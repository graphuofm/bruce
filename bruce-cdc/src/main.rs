//! Demo sidecar: subscribe to the local PG instance and keep a
//! maintained SOFTAVG view of cdc_movies fresh, printing the answer
//! after every applied transaction. Runs until idle for --idle-exit
//! seconds (0 = forever). With `--state PATH` the mirror is durable:
//! saved after every applied transaction (before the ack), and an
//! existing slot resumes FROM DISK across process death. See
//! README.md for the PG-side setup.

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Duration;

use ndarray::Array1;

use bruce_cdc::apply::{Mirror, TableMap};
use bruce_cdc::source::{ChangeSource, PgOutputSource, SlotSetup, SourceConfig};
use bruce_cdc::CdcError;

const SLOT: &str = "bruce_cdc_demo";
const EPS: f64 = 0.1;
const QUERY: [f64; 2] = [0.6, 0.8];

fn main() {
    if let Err(e) = run() {
        eprintln!("bruce-cdc: {e}");
        std::process::exit(1);
    }
}

fn arg_after(name: &str) -> Option<String> {
    std::env::args().skip_while(|a| a != name).nth(1)
}

fn run() -> Result<(), CdcError> {
    let idle_exit: u64 = arg_after("--idle-exit")
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let state: Option<PathBuf> = arg_after("--state").map(PathBuf::from);

    let cfg = SourceConfig::local_default(SLOT);
    let map = TableMap::cdc_movies();
    let mut src = PgOutputSource::connect(cfg)?;

    let mut mirror = match src.create_slot_with_snapshot()? {
        SlotSetup::CreatedSnapshotOpen { consistent_point } => {
            let (cols, rows) =
                src.snapshot_query("SELECT movie_id, genre, rating, year, e0, e1 FROM cdc_movies")?;
            src.commit_snapshot()?;
            eprintln!(
                "slot {SLOT} created at {}; snapshot {} rows",
                bruce_cdc::protocol::fmt_lsn(consistent_point),
                rows.len()
            );
            Mirror::from_snapshot(map, &cols, &rows)?
        }
        SlotSetup::AlreadyExists => match &state {
            // durable mirror: resume the existing slot from disk; the
            // durable last_lsn watermark filters the redelivered
            // prefix, exactly-once across process death
            Some(path) => {
                let m = Mirror::load(path)?;
                eprintln!(
                    "slot {SLOT} exists; resumed {} rows from {} (watermark {})",
                    m.n_rows(),
                    path.display(),
                    bruce_cdc::protocol::fmt_lsn(m.last_lsn)
                );
                m
            }
            None => {
                return Err(CdcError::Apply(format!(
                    "slot {SLOT} exists; resume with --state PATH, or drop it \
                     (SELECT pg_drop_replication_slot('{SLOT}')) to demo from scratch"
                )));
            }
        },
    };

    let x = Array1::from_vec(QUERY.to_vec());
    mirror
        .db
        .create_view("v_genre", "cdc_movies", "genre", "rating", "emb", &x, EPS)?;
    src.start()?;
    eprintln!("streaming; answers after each commit:");
    print_answer(&mut mirror)?;

    let mut idle_for = 0u64;
    loop {
        match src.next_tx(Duration::from_secs(1))? {
            Some(tx) => {
                idle_for = 0;
                let before = mirror.last_lsn;
                let n = mirror.apply_tx(&tx)?;
                if let Some(path) = &state {
                    // durable BEFORE the ack: a crash in between only
                    // redelivers txs the watermark filters
                    if mirror.last_lsn != before {
                        mirror.save(path)?;
                    }
                }
                src.ack(tx.end_lsn)?;
                if n > 0 {
                    let lag_us = bruce_cdc::protocol::now_pg_us() - tx.commit_ts_us;
                    eprintln!("applied {n} row change(s), commit->applied {lag_us} us");
                    print_answer(&mut mirror)?;
                }
            }
            None => {
                idle_for += 1;
                if idle_exit > 0 && idle_for >= idle_exit {
                    eprintln!("idle {idle_exit}s; exiting");
                    return Ok(());
                }
            }
        }
    }
}

fn print_answer(mirror: &mut Mirror) -> Result<(), CdcError> {
    let mut params = HashMap::new();
    params.insert("q".to_string(), Array1::from_vec(QUERY.to_vec()));
    let (res, planned) = mirror.db.run(
        &format!(
            "SELECT genre, SOFTAVG(rating, SIM(emb, :q), {EPS}) FROM cdc_movies GROUP BY genre"
        ),
        &params,
    )?;
    let mut pairs: Vec<_> = res.labels.iter().zip(&res.values).collect();
    pairs.sort_by(|a, b| a.0.cmp(b.0));
    let line: Vec<String> = pairs.iter().map(|(l, v)| format!("{l}={v:.6}")).collect();
    let plan_tag = match &planned.chosen {
        bruce_query::PhysicalPlan::MaintainedViewScan { view } => format!("view {view}"),
        _ => "scan".to_string(),
    };
    eprintln!("  [{plan_tag}] {}", line.join(" "));
    Ok(())
}
