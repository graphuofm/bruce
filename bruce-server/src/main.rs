//! Bruce HTTP server — exposes the F_ε retrieval primitive as a
//! network service.
//!
//! Endpoints:
//!     GET  /health                  → "ok"
//!     GET  /info                    → metadata (version, dimensions, stats)
//!     POST /facts                   → write a (k, v, owner) tuple
//!     GET  /facts/{id}              → read a fact by id (ε=0 lookup)
//!     DELETE /facts/{id}?owner=X    → owner-enforced delete
//!     POST /query/attention         → ε>0 attention query
//!     GET  /audit/root              → current Merkle root (hex)
//!     GET  /audit/length            → number of audit entries
//!     POST /audit/append            → append a custom entry
//!
//! Designed as a starting point for production deployment. Backed by
//! `bruce_core::KvMemory` + `bruce_core::MerkleAuditLog`. Single-node;
//! multi-node fan-out uses `bruce_core::distributed` (Lemma B).
//!
//! Concurrency model: a tokio `RwLock` around the mutable state lets
//! read-heavy traffic (`/info`, `/facts/:id`, `/query/attention`,
//! `/audit/*`) acquire the lock concurrently. Writes (`/facts` POST,
//! `/facts/:id` DELETE, `/audit/append`) take the exclusive write
//! lock. The dimensions `d_k`/`d_v` are immutable after construction
//! and live outside the lock to avoid spurious contention.

use std::sync::Arc;

use anyhow::Result;
use axum::{
    extract::{Path, Query, State},
    http::{Request, StatusCode},
    middleware::{self, Next},
    response::{Json, Response},
    routing::{delete, get, post},
    Router,
};
use bruce_core::{
    memory::KvMemory,
    merkle::MerkleAuditLog,
    types::{Eps, Sim},
};
use clap::Parser;
use jsonwebtoken::{decode, DecodingKey, Validation};
use ndarray::Array1;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use tracing_subscriber::EnvFilter;

#[derive(Parser, Debug)]
#[command(name = "bruce-server", version, about)]
struct Cli {
    /// Bind address.
    #[arg(long, default_value = "0.0.0.0:8080")]
    addr: String,
    /// Key dimension.
    #[arg(long, default_value_t = 128)]
    d_k: usize,
    /// Value dimension.
    #[arg(long, default_value_t = 16)]
    d_v: usize,
    /// Path to write-ahead log (JSONL). If the file exists at
    /// startup, each entry is replayed into KvMemory before serving
    /// traffic — this is the recovery path proved durable by the
    /// failover test. If empty, no WAL is written.
    #[arg(long, default_value = "")]
    wal_path: String,
    /// JWT HS256 shared secret. If non-empty, every request other
    /// than `/health` and `/metrics` must carry a valid
    /// `Authorization: Bearer <token>` header. If empty, JWT is
    /// disabled (legacy plain-HTTP behaviour). For production, set
    /// the BRUCE_JWT_SECRET environment variable instead and pass
    /// `--jwt-secret "$BRUCE_JWT_SECRET"` from the launch script.
    #[arg(long, default_value = "")]
    jwt_secret: String,
    /// Path to a PEM-encoded TLS certificate chain. Must be set
    /// together with `--tls-key` to enable HTTPS. If empty, the
    /// server serves plain HTTP.
    #[arg(long, default_value = "")]
    tls_cert: String,
    /// Path to the PEM-encoded TLS private key.
    #[arg(long, default_value = "")]
    tls_key: String,
}

/// JWT claims accepted by the auth middleware. The `sub` claim is
/// surfaced to handlers via a request extension, so that downstream
/// write/delete enforcement can verify that the request's `owner`
/// matches the token's subject (cross-tenant safety).
#[derive(Debug, Serialize, Deserialize, Clone)]
struct JwtClaims {
    /// Subject: the owner identity this token authorises.
    sub: String,
    /// Expiry (unix seconds). `jsonwebtoken` validates this against
    /// the current wall clock by default; tokens past `exp` are
    /// rejected with InvalidToken.
    exp: usize,
}

/// Shared JWT verifier state. `None` if `--jwt-secret` was empty.
#[derive(Clone)]
struct JwtState {
    key: Arc<DecodingKey>,
}

async fn auth_middleware(
    axum::extract::State(jwt): axum::extract::State<JwtState>,
    mut req: Request<axum::body::Body>,
    next: Next,
) -> Result<Response, (StatusCode, String)> {
    // Open paths: /health, /metrics.  Everything else requires a
    // Bearer token signed with the configured HS256 secret.
    let p = req.uri().path();
    if p == "/health" || p == "/ready" || p == "/metrics" {
        return Ok(next.run(req).await);
    }
    let auth = req
        .headers()
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    let token = auth
        .strip_prefix("Bearer ")
        .ok_or((StatusCode::UNAUTHORIZED, "missing Bearer token".into()))?;
    let mut validation = Validation::new(jsonwebtoken::Algorithm::HS256);
    validation.set_required_spec_claims(&["exp", "sub"]);
    validation.leeway = 0;
    let data = decode::<JwtClaims>(token, &jwt.key, &validation)
        .map_err(|e| (StatusCode::UNAUTHORIZED, format!("invalid jwt: {e}")))?;
    req.extensions_mut().insert(data.claims);
    Ok(next.run(req).await)
}

/// Application state. `d_k`/`d_v` are immutable after construction and
/// live outside the lock so handlers reading them never block. The
/// mutable `Inner` is behind a `tokio::sync::RwLock`, which permits
/// many concurrent readers and serialises writers.
#[derive(Clone)]
struct AppState {
    inner: Arc<RwLock<Inner>>,
    d_k: usize,
    d_v: usize,
    metrics: Arc<Metrics>,
}

struct Inner {
    mem: KvMemory,
    audit: MerkleAuditLog,
    /// Append-only WAL handle. `None` if `--wal-path` was empty.
    wal: Option<std::sync::Mutex<std::fs::File>>,
}

/// Atomic counters that the `/metrics` endpoint exposes in Prometheus
/// text format. Counters are bumped from request handlers without
/// holding the main RwLock.
#[derive(Default)]
struct Metrics {
    requests_total: std::sync::atomic::AtomicU64,
    writes_total: std::sync::atomic::AtomicU64,
    writes_fail_total: std::sync::atomic::AtomicU64,
    reads_total: std::sync::atomic::AtomicU64,
    reads_404_total: std::sync::atomic::AtomicU64,
    deletes_total: std::sync::atomic::AtomicU64,
    deletes_fail_total: std::sync::atomic::AtomicU64,
    queries_total: std::sync::atomic::AtomicU64,
    wal_fail_total: std::sync::atomic::AtomicU64,
    started_unix_seconds: std::sync::atomic::AtomicU64,
}

impl Metrics {
    fn bump(c: &std::sync::atomic::AtomicU64) {
        c.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    fn get(c: &std::sync::atomic::AtomicU64) -> u64 {
        c.load(std::sync::atomic::Ordering::Relaxed)
    }
}

/// One WAL record (JSONL).
#[derive(Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
enum WalRecord {
    Write {
        id: String,
        k: Vec<f64>,
        v: Vec<f64>,
        owner: String,
    },
    Delete {
        id: String,
        owner: String,
    },
}

impl Inner {
    /// Append one record to the WAL. Returns `false` on any I/O
    /// failure so callers can surface the durability loss instead of
    /// silently acking a write that never reached the log.
    fn append_wal(&self, rec: &WalRecord) -> bool {
        let Some(handle) = &self.wal else {
            return true; // WAL disabled: nothing to lose
        };
        use std::io::Write as _;
        let line = match serde_json::to_string(rec) {
            Ok(l) => l,
            Err(e) => {
                tracing::error!("WAL serialize failed: {e}");
                return false;
            }
        };
        // Recover from mutex poisoning rather than crashing the server:
        // the file handle itself is still usable.
        let mut f = match handle.lock() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        };
        let ok = f
            .write_all(line.as_bytes())
            .and_then(|()| f.write_all(b"\n"))
            .and_then(|()| f.flush());
        if let Err(e) = ok {
            tracing::error!("WAL append failed: {e}");
            return false;
        }
        true
    }
}

#[derive(Deserialize)]
struct WriteRequest {
    fact_id: String,
    k: Vec<f64>,
    v: Vec<f64>,
    owner: String,
}

#[derive(Serialize)]
struct InfoResponse {
    version: &'static str,
    d_k: usize,
    d_v: usize,
    alive: usize,
    total: usize,
    audit_len: usize,
}

#[derive(Deserialize)]
struct OwnerQuery {
    owner: String,
}

#[derive(Deserialize)]
struct QueryRequest {
    x: Vec<f64>,
    eps: f64,
    sim: String,
}

#[derive(Serialize)]
struct ReadResponse {
    k: Vec<f64>,
    v: Vec<f64>,
}

#[derive(Deserialize)]
struct AppendBody {
    payload: String,
}

async fn health() -> &'static str {
    "ok"
}

/// Readiness: verifies the shared state is acquirable (no deadlock /
/// poisoned WAL) — suitable as a k8s readinessProbe, while `/health`
/// stays a trivial liveness probe.
async fn ready(State(st): State<AppState>) -> Result<&'static str, (StatusCode, String)> {
    let _ = st.inner.read().await;
    Ok("ready")
}

async fn info(State(st): State<AppState>) -> Json<InfoResponse> {
    let g = st.inner.read().await;
    Json(InfoResponse {
        version: env!("CARGO_PKG_VERSION"),
        d_k: st.d_k,
        d_v: st.d_v,
        alive: g.mem.len_alive(),
        total: g.mem.len_total(),
        audit_len: g.audit.len(),
    })
}

async fn write_fact(
    State(st): State<AppState>,
    claims: Option<axum::Extension<JwtClaims>>,
    Json(req): Json<WriteRequest>,
) -> Result<Json<&'static str>, (StatusCode, String)> {
    Metrics::bump(&st.metrics.requests_total);
    // Cross-tenant safety: with JWT enabled, the token's subject must
    // match the owner being written. Without JWT the field is trusted
    // as-is (legacy/plaintext mode, see startup warning).
    if let Some(axum::Extension(c)) = &claims {
        if c.sub != req.owner {
            return Err((
                StatusCode::FORBIDDEN,
                format!(
                    "token subject {:?} may not write as owner {:?}",
                    c.sub, req.owner
                ),
            ));
        }
    }
    let mut g = st.inner.write().await;
    let k = Array1::from(req.k.clone());
    let v = Array1::from(req.v.clone());
    if let Err(e) = g.mem.write(&req.fact_id, k.view(), v.view(), &req.owner) {
        Metrics::bump(&st.metrics.writes_fail_total);
        return Err((StatusCode::BAD_REQUEST, e.to_string()));
    }
    g.audit
        .append(format!("WRITE {} by {}", req.fact_id, req.owner).as_bytes());
    if !g.append_wal(&WalRecord::Write {
        id: req.fact_id,
        k: req.k,
        v: req.v,
        owner: req.owner,
    }) {
        Metrics::bump(&st.metrics.wal_fail_total);
        return Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            "write applied in memory but WAL append failed; durability not guaranteed".into(),
        ));
    }
    Metrics::bump(&st.metrics.writes_total);
    Ok(Json("ok"))
}

async fn read_fact(
    State(st): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<ReadResponse>, (StatusCode, String)> {
    Metrics::bump(&st.metrics.requests_total);
    Metrics::bump(&st.metrics.reads_total);
    let g = st.inner.read().await;
    match g.mem.read_exact(&id) {
        Some((k, v)) => Ok(Json(ReadResponse {
            k: k.iter().copied().collect(),
            v: v.iter().copied().collect(),
        })),
        None => {
            Metrics::bump(&st.metrics.reads_404_total);
            Err((StatusCode::NOT_FOUND, format!("{id}: not found")))
        }
    }
}

async fn delete_fact(
    State(st): State<AppState>,
    claims: Option<axum::Extension<JwtClaims>>,
    Path(id): Path<String>,
    Query(q): Query<OwnerQuery>,
) -> Result<Json<&'static str>, (StatusCode, String)> {
    Metrics::bump(&st.metrics.requests_total);
    if let Some(axum::Extension(c)) = &claims {
        if c.sub != q.owner {
            return Err((
                StatusCode::FORBIDDEN,
                format!(
                    "token subject {:?} may not delete as owner {:?}",
                    c.sub, q.owner
                ),
            ));
        }
    }
    let mut g = st.inner.write().await;
    if let Err(e) = g.mem.delete(&id, &q.owner) {
        Metrics::bump(&st.metrics.deletes_fail_total);
        return Err((StatusCode::FORBIDDEN, e.to_string()));
    }
    g.audit
        .append(format!("DELETE {} by {}", id, q.owner).as_bytes());
    if !g.append_wal(&WalRecord::Delete { id, owner: q.owner }) {
        Metrics::bump(&st.metrics.wal_fail_total);
        return Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            "delete applied in memory but WAL append failed; durability not guaranteed".into(),
        ));
    }
    Metrics::bump(&st.metrics.deletes_total);
    Ok(Json("ok"))
}

async fn metrics_endpoint(State(st): State<AppState>) -> String {
    let m = &st.metrics;
    let started = Metrics::get(&m.started_unix_seconds);
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(started);
    let uptime = now.saturating_sub(started);
    let g = st.inner.read().await;
    let alive = g.mem.len_alive() as u64;
    let total = g.mem.len_total() as u64;
    let audit_len = g.audit.len() as u64;
    drop(g);
    format!(
        "# HELP bruce_requests_total HTTP requests handled
# TYPE bruce_requests_total counter
bruce_requests_total {req}
# HELP bruce_writes_total successful writes
# TYPE bruce_writes_total counter
bruce_writes_total {w}
# HELP bruce_writes_fail_total failed writes
# TYPE bruce_writes_fail_total counter
bruce_writes_fail_total {wf}
# HELP bruce_reads_total reads attempted
# TYPE bruce_reads_total counter
bruce_reads_total {r}
# HELP bruce_reads_404_total reads returning 404
# TYPE bruce_reads_404_total counter
bruce_reads_404_total {r404}
# HELP bruce_deletes_total successful deletes
# TYPE bruce_deletes_total counter
bruce_deletes_total {d}
# HELP bruce_deletes_fail_total failed deletes
# TYPE bruce_deletes_fail_total counter
bruce_deletes_fail_total {df}
# HELP bruce_queries_total attention queries
# TYPE bruce_queries_total counter
bruce_queries_total {q}
# HELP bruce_alive_facts current alive facts (gauge)
# TYPE bruce_alive_facts gauge
bruce_alive_facts {alive}
# HELP bruce_total_facts total facts ever written (gauge)
# TYPE bruce_total_facts gauge
bruce_total_facts {total}
# HELP bruce_audit_length audit-log entries (gauge)
# TYPE bruce_audit_length gauge
bruce_audit_length {audit_len}
# HELP bruce_wal_fail_total WAL append failures (durability loss!)
# TYPE bruce_wal_fail_total counter
bruce_wal_fail_total {walf}
# HELP bruce_uptime_seconds time since startup
# TYPE bruce_uptime_seconds counter
bruce_uptime_seconds {uptime}
",
        req = Metrics::get(&m.requests_total),
        w = Metrics::get(&m.writes_total),
        wf = Metrics::get(&m.writes_fail_total),
        r = Metrics::get(&m.reads_total),
        r404 = Metrics::get(&m.reads_404_total),
        d = Metrics::get(&m.deletes_total),
        df = Metrics::get(&m.deletes_fail_total),
        q = Metrics::get(&m.queries_total),
        walf = Metrics::get(&m.wal_fail_total),
        alive = alive,
        total = total,
        audit_len = audit_len,
        uptime = uptime,
    )
}

fn parse_sim(s: &str) -> Result<Sim, (StatusCode, String)> {
    match s {
        "dot" => Ok(Sim::Dot),
        "negsq" | "neg_squared" => Ok(Sim::NegSquared),
        "indicator" => Ok(Sim::Indicator),
        _ => Err((StatusCode::BAD_REQUEST, format!("unknown sim {s:?}"))),
    }
}

async fn query_attention(
    State(st): State<AppState>,
    Json(req): Json<QueryRequest>,
) -> Result<Json<Vec<f64>>, (StatusCode, String)> {
    Metrics::bump(&st.metrics.requests_total);
    Metrics::bump(&st.metrics.queries_total);
    let eps = Eps::new(req.eps).map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;
    let sim = parse_sim(&req.sim)?;
    let x = Array1::from(req.x);
    if x.len() != st.d_k {
        return Err((
            StatusCode::BAD_REQUEST,
            format!("expected x of dim {}, got {}", st.d_k, x.len()),
        ));
    }
    let g = st.inner.read().await;
    // hot path: KvMemory::attention_query iterates rows in place; no
    // intermediate (Array2 K, Array2 V) snapshot is allocated.
    let out = g
        .mem
        .attention_query(x.view(), eps, sim)
        .unwrap_or_else(|| Array1::<f64>::zeros(st.d_v));
    Ok(Json(out.to_vec()))
}

async fn audit_root(State(st): State<AppState>) -> Json<String> {
    let g = st.inner.read().await;
    Json(hex_encode(&g.audit.root()))
}

async fn audit_length(State(st): State<AppState>) -> Json<usize> {
    let g = st.inner.read().await;
    Json(g.audit.len())
}

async fn audit_append(State(st): State<AppState>, Json(body): Json<AppendBody>) -> Json<usize> {
    let mut g = st.inner.write().await;
    let idx = g.audit.append(body.payload.as_bytes());
    Json(idx)
}

fn hex_encode(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        use std::fmt::Write as _;
        let _ = write!(s, "{:02x}", b);
    }
    s
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    let cli = Cli::parse();
    let mut mem = KvMemory::new(cli.d_k, cli.d_v);
    let mut audit = MerkleAuditLog::new();
    let mut n_replayed_writes: usize = 0;
    let mut n_replayed_deletes: usize = 0;

    // Replay WAL if present and non-empty.
    let wal_handle = if !cli.wal_path.is_empty() {
        let p = std::path::Path::new(&cli.wal_path);
        if p.exists() {
            tracing::info!("replaying WAL from {}", cli.wal_path);
            let text = std::fs::read_to_string(p).unwrap_or_default();
            for line in text.lines() {
                let line = line.trim();
                if line.is_empty() {
                    continue;
                }
                match serde_json::from_str::<WalRecord>(line) {
                    Ok(WalRecord::Write { id, k, v, owner }) => {
                        let k_arr = Array1::from(k);
                        let v_arr = Array1::from(v);
                        if mem.write(&id, k_arr.view(), v_arr.view(), &owner).is_ok() {
                            audit.append(format!("WRITE {} by {}", id, owner).as_bytes());
                            n_replayed_writes += 1;
                        }
                    }
                    Ok(WalRecord::Delete { id, owner }) => {
                        if mem.delete(&id, &owner).is_ok() {
                            audit.append(format!("DELETE {} by {}", id, owner).as_bytes());
                            n_replayed_deletes += 1;
                        }
                    }
                    Err(e) => tracing::warn!("WAL bad line: {}", e),
                }
            }
            tracing::info!(
                "replayed {} writes + {} deletes from WAL",
                n_replayed_writes,
                n_replayed_deletes
            );
        }
        // open append-only handle for new writes
        Some(std::sync::Mutex::new(
            std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&cli.wal_path)?,
        ))
    } else {
        None
    };

    let inner = Inner {
        mem,
        audit,
        wal: wal_handle,
    };
    let metrics = Arc::new(Metrics::default());
    metrics.started_unix_seconds.store(
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0),
        std::sync::atomic::Ordering::Relaxed,
    );
    let state = AppState {
        inner: Arc::new(RwLock::new(inner)),
        d_k: cli.d_k,
        d_v: cli.d_v,
        metrics,
    };

    let mut app = Router::new()
        .route("/health", get(health))
        .route("/ready", get(ready))
        .route("/info", get(info))
        .route("/facts", post(write_fact))
        .route("/facts/:id", get(read_fact))
        .route("/facts/:id", delete(delete_fact))
        .route("/query/attention", post(query_attention))
        .route("/audit/root", get(audit_root))
        .route("/audit/length", get(audit_length))
        .route("/audit/append", post(audit_append))
        .route("/metrics", get(metrics_endpoint))
        .layer(tower_http::trace::TraceLayer::new_for_http())
        .with_state(state);

    // SECURITY-001: JWT auth (optional).  If --jwt-secret is set,
    // every request except /health and /metrics requires a Bearer
    // token signed HS256 with the same secret.
    if !cli.jwt_secret.is_empty() {
        let jwt_state = JwtState {
            key: Arc::new(DecodingKey::from_secret(cli.jwt_secret.as_bytes())),
        };
        app = app.layer(middleware::from_fn_with_state(jwt_state, auth_middleware));
        tracing::info!(
            "JWT auth enabled (HS256, {}-byte secret)",
            cli.jwt_secret.len()
        );
    } else {
        tracing::warn!("JWT auth DISABLED (--jwt-secret empty); plain HTTP only");
    }

    let addr: std::net::SocketAddr = cli
        .addr
        .parse()
        .map_err(|e| anyhow::anyhow!("bad --addr {}: {e}", cli.addr))?;
    let tls_on = !cli.tls_cert.is_empty() && !cli.tls_key.is_empty();
    tracing::info!(
        "bruce-server v{} listening on {}{}",
        env!("CARGO_PKG_VERSION"),
        cli.addr,
        if tls_on { " (TLS)" } else { " (plaintext)" }
    );
    if tls_on {
        let tls_cfg = axum_server::tls_rustls::RustlsConfig::from_pem_file(
            cli.tls_cert.clone(),
            cli.tls_key.clone(),
        )
        .await
        .map_err(|e| anyhow::anyhow!("TLS load: {e}"))?;
        let handle = axum_server::Handle::new();
        let h2 = handle.clone();
        tokio::spawn(async move {
            shutdown_signal().await;
            tracing::info!("shutdown signal received; draining connections");
            h2.graceful_shutdown(Some(std::time::Duration::from_secs(15)));
        });
        axum_server::bind_rustls(addr, tls_cfg)
            .handle(handle)
            .serve(app.into_make_service())
            .await?;
    } else {
        let listener = tokio::net::TcpListener::bind(&cli.addr).await?;
        axum::serve(listener, app)
            .with_graceful_shutdown(shutdown_signal())
            .await?;
    }
    tracing::info!("bruce-server stopped cleanly");
    Ok(())
}

/// Resolve on SIGINT (ctrl-c) or SIGTERM (docker stop / kubernetes),
/// so in-flight requests drain and the WAL is left consistent.
async fn shutdown_signal() {
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };
    #[cfg(unix)]
    let term = async {
        match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(mut s) => {
                s.recv().await;
            }
            Err(_) => std::future::pending::<()>().await,
        }
    };
    #[cfg(not(unix))]
    let term = std::future::pending::<()>();
    tokio::select! {
        _ = ctrl_c => {},
        _ = term => {},
    }
}
