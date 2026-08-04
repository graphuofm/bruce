//! Watch the temperature-aware planner decide.
//!
//!     cargo run -p bruce-query --example explain_demo
//!
//! One synthetic table, one query shape, and the knobs that flip the
//! plan: the temperature, the declared error budget, and a maintained
//! view.

use std::collections::HashMap;

use ndarray::{Array1, Array2};

use bruce_query::{Column, Database, Table};

fn main() {
    let n = 100_000;
    let groups = ["A", "B", "C", "D"];
    let sims: Vec<f64> = (0..n)
        .map(|i| 1.0 - 2.0 * (i as f64) / (n as f64))
        .collect();
    let mut keys = Array2::<f64>::zeros((n, 64));
    for i in 0..n {
        keys[(i, 0)] = sims[i];
    }
    let mut state = 999u64;
    let vals: Vec<f64> = (0..n)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state % 10_000) as f64 / 1000.0
        })
        .collect();
    let mut t = Table::default();
    t.columns.insert(
        "genre".into(),
        Column::DictU32 {
            codes: (0..n).map(|i| (i % 4) as u32).collect(),
            dict: groups.iter().map(|s| s.to_string()).collect(),
        },
    );
    t.columns.insert("rating".into(), Column::ScalarF64(vals));
    t.columns.insert(
        "id".into(),
        Column::ScalarF64((0..n).map(|i| i as f64).collect()),
    );
    t.columns.insert("emb".into(), Column::KeyF64(keys));

    let mut db = Database::new();
    db.register("movies", t);
    let mut params = HashMap::new();
    let mut q = vec![0.0; 64];
    q[0] = 1.0;
    params.insert("q".to_string(), Array1::from(q.clone()));

    let show = |title: &str, db: &mut Database, sql: &str, params: &_| {
        println!("---- {title}\n     {sql}");
        match db.run(sql, params) {
            Ok((_, planned)) => println!("{}", planned.explain()),
            Err(e) => println!("error: {e}"),
        }
    };

    show(
        "exact soft aggregate (no budget: only exact plans enumerate)",
        &mut db,
        "SELECT genre, SOFTAVG(rating, SIM(emb, :q), 0.02) FROM movies GROUP BY genre",
        &params,
    );
    show(
        "sharp + budget: the sketch certifies a small k*, contract wins",
        &mut db,
        "SELECT genre, SOFTAVG(rating, SIM(emb, :q), 0.02, 0.05) FROM movies GROUP BY genre",
        &params,
    );
    show(
        "diffuse + budget: near-uniform weights, contract buys nothing",
        &mut db,
        "SELECT genre, SOFTAVG(rating, SIM(emb, :q), 1.0, 0.05) FROM movies GROUP BY genre",
        &params,
    );
    show(
        "super-sharp + budget: sketch is resolution-limited, refuses",
        &mut db,
        "SELECT genre, SOFTAVG(rating, SIM(emb, :q), 0.0001, 0.05) FROM movies GROUP BY genre",
        &params,
    );
    show(
        "eps = INF: R3 degenerates to the exact uniform average",
        &mut db,
        "SELECT genre, SOFTAVG(rating, SIM(emb, :q), INF) FROM movies GROUP BY genre",
        &params,
    );

    let x = Array1::from(q);
    db.create_view("v_genre", "movies", "genre", "rating", "emb", &x, 0.02)
        .unwrap();
    show(
        "same query with a maintained view registered: O(groups) read",
        &mut db,
        "SELECT genre, SOFTAVG(rating, SIM(emb, :q), 0.02) FROM movies GROUP BY genre",
        &params,
    );
}
