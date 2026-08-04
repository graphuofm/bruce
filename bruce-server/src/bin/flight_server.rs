//! bruce-flight-server — the Arrow Flight surface (SQL-in-ticket v1).
//!
//! Usage:
//!
//! ```text
//! bruce-flight-server --flight-addr 127.0.0.1:0 \
//!     --parquet movies=/data/movies.parquet \
//!     --key movies:emb=/data/emb.npy
//! ```
//!
//! Loads each `--parquet name=path` table via the bruce-query Arrow
//! ingest, attaches each `--key table:col=npy_path` embedding matrix
//! at its stored dtype (f32 stays f32 — the M2 precision contract),
//! registers everything into one `Database`, and serves Arrow Flight.
//!
//! Startup prints `FLIGHT_PORT <port>` on stdout once bound — with
//! port 0 the OS assigns an ephemeral port and this line is how
//! scripts learn it (logs go to stderr, keeping stdout parseable).
//! Graceful shutdown on ctrl-c / SIGTERM.

use std::collections::HashMap;
use std::io::Write as _;

use anyhow::{anyhow, Context, Result};
use bruce_query::{Database, Table};
use bruce_server::flight::{serve_with_shutdown, BruceFlightService};
use bruce_server::npy::{read_npy_2d, NpyMatrix};
use clap::Parser;
use tracing_subscriber::EnvFilter;

#[derive(Parser, Debug)]
#[command(name = "bruce-flight-server", version, about)]
struct Cli {
    /// Bind address. Port 0 = OS-assigned ephemeral port; the bound
    /// port is printed on stdout as `FLIGHT_PORT <port>`.
    #[arg(long, default_value = "127.0.0.1:0")]
    flight_addr: String,
    /// Table to load from Parquet, as name=path. Repeatable.
    #[arg(long = "parquet", value_name = "NAME=PATH")]
    parquet: Vec<String>,
    /// Key (embedding) matrix to attach, as table:col=npy_path.
    /// The npy dtype decides the storage dtype. Repeatable.
    #[arg(long = "key", value_name = "TABLE:COL=NPY_PATH")]
    key: Vec<String>,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Logs on stderr: stdout is the machine-readable channel
    // (FLIGHT_PORT line).
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .with_writer(std::io::stderr)
        .init();
    let cli = Cli::parse();

    let mut tables: HashMap<String, Table> = HashMap::new();
    for spec in &cli.parquet {
        let (name, path) = spec
            .split_once('=')
            .ok_or_else(|| anyhow!("--parquet expects NAME=PATH, got {spec:?}"))?;
        let t0 = std::time::Instant::now();
        let table = Table::from_parquet(path).map_err(|e| anyhow!("load {path}: {e}"))?;
        let rows = table.columns.values().next().map(|c| c.len()).unwrap_or(0);
        tracing::info!(
            "loaded table {name:?}: {rows} rows from {path} in {:.2}s",
            t0.elapsed().as_secs_f64()
        );
        tables.insert(name.to_string(), table);
    }
    for spec in &cli.key {
        let (tc, path) = spec
            .split_once('=')
            .ok_or_else(|| anyhow!("--key expects TABLE:COL=NPY_PATH, got {spec:?}"))?;
        let (tname, col) = tc
            .split_once(':')
            .ok_or_else(|| anyhow!("--key expects TABLE:COL=NPY_PATH, got {spec:?}"))?;
        let table = tables.get_mut(tname).ok_or_else(|| {
            anyhow!("--key names unknown table {tname:?}; load it with --parquet first")
        })?;
        let t0 = std::time::Instant::now();
        let m = read_npy_2d(path).map_err(|e| anyhow!("--key {spec}: {e}"))?;
        let (rows, d) = m.shape();
        let dtype = match &m {
            NpyMatrix::F32(_) => "f32",
            NpyMatrix::F64(_) => "f64",
        };
        match m {
            NpyMatrix::F32(a) => table.attach_key_f32(col, a),
            NpyMatrix::F64(a) => table.attach_key_f64(col, a),
        }
        .map_err(|e| anyhow!("--key {spec}: {e}"))?;
        tracing::info!(
            "attached key {tname}:{col}: {rows} x {d} {dtype} from {path} in {:.2}s",
            t0.elapsed().as_secs_f64()
        );
    }

    let mut db = Database::new();
    for (name, table) in tables {
        db.register(&name, table); // collects stats
    }

    let listener = tokio::net::TcpListener::bind(&cli.flight_addr)
        .await
        .with_context(|| format!("bind {}", cli.flight_addr))?;
    let addr = listener.local_addr()?;
    // Machine-readable line first (keep stable: scripts parse it),
    // then the human one.
    println!("FLIGHT_PORT {}", addr.port());
    println!(
        "bruce-flight-server v{} listening on {addr} (Arrow Flight, SQL-in-ticket)",
        env!("CARGO_PKG_VERSION")
    );
    std::io::stdout().flush()?;

    serve_with_shutdown(BruceFlightService::new(db), listener, shutdown_signal()).await?;
    tracing::info!("bruce-flight-server stopped cleanly");
    Ok(())
}

/// Resolve on SIGINT (ctrl-c) or SIGTERM (docker stop / kubernetes),
/// so in-flight requests drain before exit.
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
