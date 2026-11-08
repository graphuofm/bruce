//! ladder_replay — the PostgreSQL call sequences of the softmax ladder, replayed
//! outside the server over the same Rust arithmetic the extensions link.
//!
//! NOT an engine run (no latency, no executor): it answers "what values would
//! PostgreSQL deliver", given PG's documented call order:
//!
//! window (ROWS BETWEEN k PRECEDING AND CURRENT ROW, nodeWindowAgg.c
//! eval_windowaggregates): when the frame head advances, MINVFUNC is called for
//! the row that left; if it returns NULL the aggregate restarts (fresh state,
//! MSFUNC over every frame row); then MSFUNC for the new row; then MFINALFUNC.
//!   S2  no MINVFUNC (softavg_fwd): restart every frame -> ScalarAcc fold of the frame
//!   S3  MINVFUNC = PlainInv::remove, never NULL
//!   S4  MINVFUNC = InvAcc::remove, NULL when !is_exact (bruce_pg 0.2.0 softavg_minvfunc)
//!
//! ivm: one grouped view under a statement stream, mirroring the plpgsql
//! trigger of bruce_pg softavg_ivm_create: per statement, DELETE rows are
//! removed from their group's state, INSERT rows are folded into a delta state
//! and merged, UPDATE = remove old + merge new; then
//!   S4  groups whose InvAcc is inexact are recomputed from the base rows
//!   S3  nothing is recomputed (plain removal, no certificate)
//!   S2  every group a DELETE/UPDATE touched is recomputed (no removal at all)
//! A group whose row count reaches 0 is dropped (as the trigger's DELETE ... cnt = 0).
//!
//! usage:
//!   ladder_replay window <in.bin> <k> <eps> <out.bin>
//!       in.bin: u64 n, f64 s[n], f64 v[n]; out.bin: f64 s2[n], s3[n], s4[n] (NaN = NULL)
//!       stdout: JSON {punts, restart_rows}
//!   ladder_replay ivm <rows.bin> <init.bin> <ops.bin> <eps> <v_scale> <checkpoint_every_stmts>
//!       rows.bin: u64 n, i32 facet[n], f64 v[n], f64 s[n]   (indexed by id)
//!       init.bin: u64 m, u32 id[m]
//!       ops.bin:  u64 nstmt, then per statement u8 kind (0 ins, 1 del, 2 upd), u32 cnt,
//!                 cnt x (u32 id [, f64 v, f64 s for upd])
//!       stdout: JSON per arm

use ladder_core::PlainInv;
use softagg_core::{InvAcc, ScalarAcc};
use std::collections::{BTreeSet, HashMap, HashSet};
use std::fs;
use std::io::Write;

struct Rd<'a> {
    b: &'a [u8],
    p: usize,
}
impl<'a> Rd<'a> {
    fn u8(&mut self) -> u8 {
        self.p += 1;
        self.b[self.p - 1]
    }
    fn u32(&mut self) -> u32 {
        self.p += 4;
        u32::from_le_bytes(self.b[self.p - 4..self.p].try_into().unwrap())
    }
    fn u64(&mut self) -> u64 {
        self.p += 8;
        u64::from_le_bytes(self.b[self.p - 8..self.p].try_into().unwrap())
    }
    fn i32(&mut self) -> i32 {
        self.u32() as i32
    }
    fn f64(&mut self) -> f64 {
        f64::from_bits(self.u64())
    }
}

fn jnum(x: f64) -> String {
    if x.is_finite() {
        format!("{x:e}")
    } else {
        "null".into()
    }
}

fn opt(x: Option<f64>) -> f64 {
    x.unwrap_or(f64::NAN)
}

fn window(args: &[String]) {
    let b = fs::read(&args[0]).unwrap();
    let k: usize = args[1].parse().unwrap();
    let eps: f64 = args[2].parse().unwrap();
    let mut r = Rd { b: &b, p: 0 };
    let n = r.u64() as usize;
    let s: Vec<f64> = (0..n).map(|_| r.f64()).collect();
    let v: Vec<f64> = (0..n).map(|_| r.f64()).collect();
    let (mut o2, mut o3, mut o4) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    let mut s3 = PlainInv::new();
    let mut s4 = InvAcc::new();
    let (mut punts, mut restart_rows) = (0u64, 0u64);
    for i in 0..n {
        let head = i.saturating_sub(k);
        // S2: restart every frame
        let mut a = ScalarAcc::new();
        for j in head..=i {
            a.absorb(s[j], v[j], eps);
        }
        o2[i] = opt(a.finalize());
        // S3 / S4: inverse for the row that left, then the new row
        if i > k {
            let j = i - k - 1;
            s3.remove(s[j], v[j], eps);
            s4.remove(s[j], v[j], eps);
        }
        s3.absorb(s[i], v[i], eps);
        if !s4.is_exact() {
            punts += 1;
            s4 = InvAcc::new();
            for j in head..=i {
                s4.absorb(s[j], v[j], eps);
            }
            restart_rows += (i + 1 - head) as u64;
        } else {
            s4.absorb(s[i], v[i], eps);
        }
        o3[i] = opt(s3.finalize());
        if !s4.is_exact() {
            // a fresh fold is always exact; an absorb never removes, so this
            // would be a core bug: record it loudly
            eprintln!("S4 inexact after absorb at row {i}");
        }
        o4[i] = opt(s4.finalize());
    }
    let mut f = fs::File::create(&args[3]).unwrap();
    for arr in [&o2, &o3, &o4] {
        let bytes: Vec<u8> = arr.iter().flat_map(|x| x.to_le_bytes()).collect();
        f.write_all(&bytes).unwrap();
    }
    println!("{{\"punts\":{punts},\"restart_rows\":{restart_rows}}}");
}

#[derive(Clone, Copy, PartialEq)]
enum Arm {
    S2,
    S3,
    S4,
}

/// One arm's per-group states.
struct View {
    arm: Arm,
    inv: HashMap<i32, InvAcc>,
    plain: HashMap<i32, PlainInv>,
    cnt: HashMap<i32, u64>,
    recomputed: u64,
    refold_rows: u64,
}

fn fold_inv(ids: &BTreeSet<u32>, cur: &[(f64, f64)], eps: f64) -> InvAcc {
    let mut a = InvAcc::new();
    for &i in ids {
        let (v, s) = cur[i as usize];
        a.absorb(s, v, eps);
    }
    a
}

fn fold_plain(ids: &BTreeSet<u32>, cur: &[(f64, f64)], eps: f64) -> PlainInv {
    let mut a = PlainInv::new();
    for &i in ids {
        let (v, s) = cur[i as usize];
        a.absorb(s, v, eps);
    }
    a
}

impl View {
    fn new(arm: Arm, live: &HashMap<i32, BTreeSet<u32>>, cur: &[(f64, f64)], eps: f64) -> Self {
        let mut w = View { arm, inv: HashMap::new(), plain: HashMap::new(), cnt: HashMap::new(), recomputed: 0, refold_rows: 0 };
        for (&g, ids) in live {
            w.cnt.insert(g, ids.len() as u64);
            match arm {
                Arm::S4 => {
                    w.inv.insert(g, fold_inv(ids, cur, eps));
                }
                _ => {
                    w.plain.insert(g, fold_plain(ids, cur, eps));
                }
            }
        }
        w
    }

    fn value(&self, g: i32) -> Option<f64> {
        match self.arm {
            Arm::S4 => {
                let a = self.inv.get(&g)?;
                assert!(a.is_exact(), "S4 state read while inexact");
                a.finalize()
            }
            _ => self.plain.get(&g)?.finalize(),
        }
    }

    fn recompute(&mut self, g: i32, live: &HashMap<i32, BTreeSet<u32>>, cur: &[(f64, f64)], eps: f64) {
        let empty = BTreeSet::new();
        let ids = live.get(&g).unwrap_or(&empty);
        self.recomputed += 1;
        self.refold_rows += ids.len() as u64;
        match self.arm {
            Arm::S4 => {
                self.inv.insert(g, fold_inv(ids, cur, eps));
            }
            _ => {
                self.plain.insert(g, fold_plain(ids, cur, eps));
            }
        }
    }

    /// One statement: `dels` = (g, v, s) of removed rows, `ins` = (g, v, s) of added rows,
    /// in statement order. Base membership (`live`, `cur`) is already updated.
    fn apply(&mut self, dels: &[(i32, f64, f64)], ins: &[(i32, f64, f64)], live: &HashMap<i32, BTreeSet<u32>>,
             cur: &[(f64, f64)], eps: f64) {
        let mut touched: Vec<i32> = Vec::new();
        let mut del_groups: HashSet<i32> = HashSet::new();
        for &(g, v, s) in dels {
            touched.push(g);
            del_groups.insert(g);
            *self.cnt.get_mut(&g).unwrap() -= 1;
            match self.arm {
                Arm::S4 => {
                    if let Some(a) = self.inv.get_mut(&g) {
                        a.remove(s, v, eps);
                    }
                }
                Arm::S3 => {
                    if let Some(a) = self.plain.get_mut(&g) {
                        a.remove(s, v, eps);
                    }
                }
                Arm::S2 => {}
            }
        }
        // insert deltas, one per group, folded in statement order, then merged
        let mut delta_inv: Vec<(i32, InvAcc)> = Vec::new();
        let mut delta_plain: Vec<(i32, PlainInv)> = Vec::new();
        for &(g, v, s) in ins {
            touched.push(g);
            *self.cnt.entry(g).or_insert(0) += 1;
            match self.arm {
                Arm::S4 => match delta_inv.iter_mut().find(|x| x.0 == g) {
                    Some(x) => x.1.absorb(s, v, eps),
                    None => {
                        let mut a = InvAcc::new();
                        a.absorb(s, v, eps);
                        delta_inv.push((g, a));
                    }
                },
                _ => match delta_plain.iter_mut().find(|x| x.0 == g) {
                    Some(x) => x.1.absorb(s, v, eps),
                    None => {
                        let mut a = PlainInv::new();
                        a.absorb(s, v, eps);
                        delta_plain.push((g, a));
                    }
                },
            }
        }
        for (g, d) in delta_inv {
            self.inv.entry(g).or_insert_with(InvAcc::new).merge(&d, eps);
        }
        for (g, d) in delta_plain {
            self.plain.entry(g).or_insert_with(PlainInv::new).merge(&d, eps);
        }
        touched.sort();
        touched.dedup();
        for &g in &touched {
            if self.cnt[&g] == 0 {
                self.cnt.remove(&g);
                self.inv.remove(&g);
                self.plain.remove(&g);
            }
        }
        for &g in &touched {
            if !self.cnt.contains_key(&g) {
                continue;
            }
            let redo = match self.arm {
                Arm::S4 => !self.inv[&g].is_exact(),
                Arm::S3 => false,
                Arm::S2 => del_groups.contains(&g),
            };
            if redo {
                self.recompute(g, live, cur, eps);
            }
        }
    }
}

fn fresh(g: i32, live: &HashMap<i32, BTreeSet<u32>>, cur: &[(f64, f64)], eps: f64) -> Option<f64> {
    let mut a = ScalarAcc::new();
    for &i in live.get(&g)? {
        let (v, s) = cur[i as usize];
        a.absorb(s, v, eps);
    }
    a.finalize()
}

#[derive(Default)]
struct Drift {
    max_abs: f64,
    nonfinite: u64,
    null_mismatch: u64,
    wrong: u64, // > 1e-9 * v_scale, or non-finite / NULL where fresh has a value
    checked: u64,
}

impl Drift {
    fn add(&mut self, got: Option<f64>, want: Option<f64>, v_scale: f64) {
        self.checked += 1;
        match (got, want) {
            (Some(x), Some(y)) => {
                if !x.is_finite() {
                    self.nonfinite += 1;
                    self.wrong += 1;
                } else {
                    let d = (x - y).abs() / v_scale;
                    if d > self.max_abs {
                        self.max_abs = d;
                    }
                    if d > 1e-9 {
                        self.wrong += 1;
                    }
                }
            }
            (None, None) => {}
            _ => {
                self.null_mismatch += 1;
                self.wrong += 1;
            }
        }
    }
    fn json(&self) -> String {
        format!(
            "{{\"max_abs_err_over_vscale\":{},\"nonfinite\":{},\"null_mismatch\":{},\"wrong\":{},\"checked\":{}}}",
            jnum(self.max_abs), self.nonfinite, self.null_mismatch, self.wrong, self.checked
        )
    }
}

fn ivm(args: &[String]) {
    let rb = fs::read(&args[0]).unwrap();
    let ib = fs::read(&args[1]).unwrap();
    let ob = fs::read(&args[2]).unwrap();
    let eps: f64 = args[3].parse().unwrap();
    let v_scale: f64 = args[4].parse().unwrap();
    let cp_every: usize = args[5].parse().unwrap();
    let mut r = Rd { b: &rb, p: 0 };
    let n = r.u64() as usize;
    let facet: Vec<i32> = (0..n).map(|_| r.i32()).collect();
    let vv: Vec<f64> = (0..n).map(|_| r.f64()).collect();
    let ss: Vec<f64> = (0..n).map(|_| r.f64()).collect();
    let mut cur: Vec<(f64, f64)> = vv.into_iter().zip(ss).collect();
    let mut r = Rd { b: &ib, p: 0 };
    let m = r.u64() as usize;
    let mut live: HashMap<i32, BTreeSet<u32>> = HashMap::new();
    for _ in 0..m {
        let i = r.u32();
        live.entry(facet[i as usize]).or_default().insert(i);
    }
    let mut views: Vec<View> = [Arm::S2, Arm::S3, Arm::S4].iter().map(|&a| View::new(a, &live, &cur, eps)).collect();
    let mut per_stmt: Vec<Drift> = (0..3).map(|_| Drift::default()).collect();
    let mut cps: Vec<String> = Vec::new();
    let mut first_wrong_stmt: Vec<i64> = vec![-1; 3];
    let mut r = Rd { b: &ob, p: 0 };
    let nstmt = r.u64() as usize;
    for st in 0..nstmt {
        let kind = r.u8();
        let cnt = r.u32() as usize;
        let (mut dels, mut ins) = (Vec::new(), Vec::new());
        for _ in 0..cnt {
            let i = r.u32() as usize;
            let g = facet[i];
            match kind {
                0 => {
                    live.entry(g).or_default().insert(i as u32);
                    ins.push((g, cur[i].0, cur[i].1));
                }
                1 => {
                    let set = live.get_mut(&g).unwrap();
                    assert!(set.remove(&(i as u32)), "delete of a dead row {i}");
                    if set.is_empty() {
                        live.remove(&g);
                    }
                    dels.push((g, cur[i].0, cur[i].1));
                }
                _ => {
                    let (nv, ns) = (r.f64(), r.f64());
                    dels.push((g, cur[i].0, cur[i].1));
                    cur[i] = (nv, ns);
                    ins.push((g, nv, ns));
                }
            }
        }
        let mut tg: Vec<i32> = dels.iter().chain(ins.iter()).map(|x| x.0).collect();
        tg.sort();
        tg.dedup();
        for (a, w) in views.iter_mut().enumerate() {
            w.apply(&dels, &ins, &live, &cur, eps);
            // per-statement drift on the touched groups
            for &g in &tg {
                let want = fresh(g, &live, &cur, eps);
                let before = per_stmt[a].wrong;
                per_stmt[a].add(w.value(g), want, v_scale);
                if per_stmt[a].wrong > before && first_wrong_stmt[a] < 0 {
                    first_wrong_stmt[a] = st as i64;
                }
            }
        }
        if (st + 1) % cp_every == 0 || st + 1 == nstmt {
            let mut parts = Vec::new();
            for w in views.iter() {
                let mut d = Drift::default();
                let mut gs: Vec<i32> = live.keys().copied().chain(w.cnt.keys().copied()).collect();
                gs.sort();
                gs.dedup();
                for g in gs {
                    d.add(w.value(g), fresh(g, &live, &cur, eps), v_scale);
                }
                parts.push(format!("\"{}\":{}", ["S2", "S3", "S4"][w.arm as usize], d.json()));
            }
            cps.push(format!("{{\"after_statements\":{},\"live_rows\":{},{}}}", st + 1,
                             live.values().map(|x| x.len()).sum::<usize>(), parts.join(",")));
        }
    }
    let arms: Vec<String> = views
        .iter()
        .enumerate()
        .map(|(a, w)| {
            format!(
                "\"{}\":{{\"groups_recomputed\":{},\"refold_rows\":{},\"per_statement_touched\":{},\"first_wrong_statement\":{}}}",
                ["S2", "S3", "S4"][a], w.recomputed, w.refold_rows, per_stmt[a].json(), first_wrong_stmt[a]
            )
        })
        .collect();
    println!("{{\"statements\":{nstmt},\"arms\":{{{}}},\"checkpoints\":[{}]}}", arms.join(","), cps.join(","));
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    match a.get(1).map(|s| s.as_str()) {
        Some("window") => window(&a[2..]),
        Some("ivm") => ivm(&a[2..]),
        _ => {
            eprintln!("usage: ladder_replay window|ivm ...");
            std::process::exit(2);
        }
    }
}
