//! Workstream 5 — SQL frontend fuzz.
//!
//! (a) 500 structurally-random VALID queries over the schema
//!     `{g: dict, v, y: scalar, k: key}` (random eps literals including
//!     0, huge, tiny, saturating, INF; random predicates; random
//!     whitespace and keyword case): `parse_query` must succeed, the
//!     parsed structure must match the generator's intent, and
//!     re-lowering through `optimize` must not panic.
//! (b) 500 MUTATED/malformed strings (token deletion / swap /
//!     injection incl. non-ASCII junk, unterminated literals, deep
//!     nesting, truncation): `parse_query` must return `Err` and must
//!     NEVER panic — proven by `catch_unwind` around every call.
//!
//! Everything is seeded and deterministic: a failure reproduces from
//! its printed seed alone.

use std::panic::{catch_unwind, AssertUnwindSafe};

use bruce_query::{optimize, parse_query, LogicalPlan};

// ------------------------------------------------------------------
// Deterministic RNG (xorshift64*): no external dependency, no global
// state, identical sequences on every platform.
// ------------------------------------------------------------------

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
    fn coin(&mut self) -> bool {
        self.next() & 1 == 0
    }
    fn pick<'a>(&mut self, xs: &'a [&'a str]) -> &'a str {
        xs[self.below(xs.len())]
    }
}

fn rand_case(rng: &mut Rng, s: &str) -> String {
    s.chars()
        .map(|c| {
            if rng.coin() {
                c.to_ascii_uppercase()
            } else {
                c.to_ascii_lowercase()
            }
        })
        .collect()
}

// ------------------------------------------------------------------
// Valid-query generator over the fixed fuzz schema
//   g: dict group column, v/y: scalar columns, k: key column, t: table
// ------------------------------------------------------------------

/// eps literals: 0, tiny, huge, subnormal-adjacent, E-notation, and
/// `1e999` which Rust float parsing saturates to +inf — pinned below
/// as the uniform-mean endpoint (same saturation in `num_of`).
const EPS_LITS: [&str; 14] = [
    "0",
    "0.0",
    "0.001",
    "0.3",
    "1",
    "2.5",
    "17.25",
    "1000000",
    "1e-9",
    "1e9",
    "1e308",
    "1e-308",
    "1e999",
    "0.00000000000000001",
];

const BUDGET_LITS: [&str; 4] = ["0.01", "0.05", "0.5", "1"];
const WHERE_NUMS: [&str; 6] = ["0", "1.5", "2000", "0.25", "123456789.75", "3"];

struct ValidQuery {
    tokens: Vec<String>,
    want_eps: f64,
    want_kind: &'static str, // lowercased sim function name
}

/// SOFTAVG(v, <sim>(k, :q), <eps>[, <budget>]) as a token vector.
fn softavg_tokens(rng: &mut Rng) -> (Vec<String>, f64, &'static str) {
    let simfn = ["sim", "dot", "negsq", "indicator"][rng.below(4)];
    let (eps_lit, want_eps) = if rng.below(5) == 0 {
        let lit = if rng.coin() { "INF" } else { "INFINITY" };
        (rand_case(rng, lit), f64::INFINITY)
    } else {
        let lit = rng.pick(&EPS_LITS);
        (lit.to_string(), lit.parse::<f64>().unwrap())
    };
    let mut t: Vec<String> = vec![
        rand_case(rng, "softavg"),
        "(".into(),
        "v".into(),
        ",".into(),
        rand_case(rng, simfn),
        "(".into(),
        "k".into(),
        ",".into(),
        ":q".into(),
        ")".into(),
        ",".into(),
        eps_lit,
    ];
    if rng.below(3) == 0 {
        t.push(",".into());
        t.push(rng.pick(&BUDGET_LITS).to_string());
    }
    t.push(")".into());
    (t, want_eps, simfn)
}

fn gen_valid(rng: &mut Rng, allow_extra: bool) -> ValidQuery {
    let (sa, want_eps, want_kind) = softavg_tokens(rng);
    let mut t: Vec<String> = vec![rand_case(rng, "select")];
    if rng.coin() {
        t.push("g".into());
        if allow_extra && rng.below(4) == 0 {
            t.push(",".into());
            t.push("y".into());
        }
        t.push(",".into());
        t.extend(sa);
    } else {
        t.extend(sa);
        t.push(",".into());
        t.push("g".into());
    }
    t.push(rand_case(rng, "from"));
    t.push("t".into());
    if rng.below(3) > 0 {
        t.push(rand_case(rng, "where"));
        t.push(if rng.coin() { "y" } else { "v" }.into());
        t.push(if rng.coin() { ">=" } else { "=" }.into());
        t.push(rng.pick(&WHERE_NUMS).to_string());
    }
    t.push(rand_case(rng, "group"));
    t.push(rand_case(rng, "by"));
    t.push("g".into());
    ValidQuery {
        tokens: t,
        want_eps,
        want_kind,
    }
}

/// Join tokens with random whitespace (space/tab/newline/CR); returns
/// the SQL string and each token's byte offset (used by truncation).
fn join(rng: &mut Rng, tokens: &[String]) -> (String, Vec<usize>) {
    const WS: [char; 4] = [' ', '\t', '\n', '\r'];
    let alnum = |c: char| c.is_ascii_alphanumeric() || c == '_';
    let mut s = String::new();
    let mut starts = Vec::with_capacity(tokens.len());
    for (i, tok) in tokens.iter().enumerate() {
        if i > 0 {
            let prev = tokens[i - 1].chars().last().unwrap();
            let next = tok.chars().next().unwrap();
            let n_ws = if alnum(prev) && alnum(next) {
                1 + rng.below(3)
            } else {
                rng.below(3)
            };
            for _ in 0..n_ws {
                s.push(WS[rng.below(4)]);
            }
        }
        starts.push(s.len());
        s.push_str(tok);
    }
    (s, starts)
}

// ------------------------------------------------------------------
// (a) valid corpus
// ------------------------------------------------------------------

#[test]
fn fuzz_500_valid_queries_parse_and_relower() {
    for seed in 0..500u64 {
        let mut rng = Rng::new(seed);
        let q = gen_valid(&mut rng, true);
        let (sql, _) = join(&mut rng, &q.tokens);
        let sql = if rng.below(4) == 0 {
            format!("{sql};")
        } else {
            sql
        };

        let parsed = catch_unwind(AssertUnwindSafe(|| parse_query(&sql)))
            .unwrap_or_else(|_| panic!("seed {seed}: parse_query PANICKED on VALID input {sql:?}"));
        let plan =
            parsed.unwrap_or_else(|e| panic!("seed {seed}: valid query rejected ({e}):\n{sql:?}"));

        let LogicalPlan::SoftAgg {
            group_col,
            val_col,
            score,
            eps,
            ..
        } = &plan
        else {
            panic!("seed {seed}: expected SoftAgg root, got {plan:?}");
        };
        assert_eq!(group_col, "g", "seed {seed}");
        assert_eq!(val_col, "v", "seed {seed}");
        assert_eq!(score.key_col, "k", "seed {seed}");
        assert_eq!(score.param, "q", "seed {seed}");
        let kind = format!("{:?}", score.kind).to_ascii_lowercase();
        let want_kind = match q.want_kind {
            "sim" | "dot" => "dot",
            other => other,
        };
        assert_eq!(kind, want_kind, "seed {seed}");
        assert_eq!(
            eps.to_bits(),
            q.want_eps.to_bits(),
            "seed {seed}: eps {} != expected {}",
            eps,
            q.want_eps
        );

        // re-lowering must not panic, and R3 must fire exactly on inf
        let opt = catch_unwind(AssertUnwindSafe(|| optimize(plan.clone())))
            .unwrap_or_else(|_| panic!("seed {seed}: optimize PANICKED on {sql:?}"));
        match (&opt, q.want_eps.is_infinite()) {
            (LogicalPlan::PlainGroupAvg { .. }, true) => {}
            (LogicalPlan::SoftAgg { .. }, false) => {}
            _ => panic!(
                "seed {seed}: wrong optimized root for eps={}: {opt:?}",
                q.want_eps
            ),
        }
    }
}

/// Pinned semantics: an eps literal beyond f64 range (`1e999`)
/// saturates to +inf under Rust float parsing (`num_of` uses
/// `str::parse::<f64>`), so the query degenerates to the exact
/// uniform-mean endpoint via R3 — it is NOT a parse error.
#[test]
fn eps_literal_overflow_saturates_to_uniform_mean_endpoint() {
    let plan = parse_query("SELECT g, SOFTAVG(v, SIM(k, :q), 1e999) FROM t GROUP BY g").unwrap();
    let LogicalPlan::SoftAgg { eps, .. } = &plan else {
        panic!("expected SoftAgg");
    };
    assert!(eps.is_infinite() && eps.is_sign_positive());
    assert!(matches!(optimize(plan), LogicalPlan::PlainGroupAvg { .. }));
}

// ------------------------------------------------------------------
// (b) malformed corpus
// ------------------------------------------------------------------

/// Junk tokens that can never appear in a valid query at any position:
/// unterminated quote/ident starts, unbalanced parens, an unterminated
/// block comment, and non-alphabetic unicode (math symbols, emoji,
/// NUL, an RTL-override control char) which GenericDialect rejects.
const JUNK: [&str; 12] = [
    "'",
    "\"",
    "`",
    "((",
    "))",
    "/*",
    "\u{2211}",           // ∑ (Sm, not alphabetic -> not an identifier char)
    "\u{1F4A5}",          // 💥
    "\u{0000}",           // NUL
    "\u{202E}",           // RIGHT-TO-LEFT OVERRIDE
    "\u{00AC}\u{00AC}",   // ¬¬
    "\u{1F980}\u{1F980}", // 🦀🦀
];

/// Every mutation operator is constructed to guarantee the result is
/// malformed for THIS grammar (a complete `GROUP BY <ident>` plus a
/// well-formed SOFTAVG projection is required for `Ok`), so the test
/// can assert `Err` — not merely "no panic" — on all 500 mutants.
fn mutate(rng: &mut Rng, tokens: &[String]) -> String {
    let find = |t: &str| tokens.iter().position(|x| x.eq_ignore_ascii_case(t));
    match rng.below(8) {
        0 => {
            // delete a structurally required token
            let mut cands: Vec<usize> = Vec::new();
            for t in ["select", "from", "group", "by"] {
                if let Some(i) = find(t) {
                    cands.push(i);
                }
            }
            for (i, t) in tokens.iter().enumerate() {
                if t == "(" || t == ")" || t == "," {
                    cands.push(i);
                }
            }
            let i = cands[rng.below(cands.len())];
            let mut v = tokens.to_vec();
            v.remove(i);
            join(rng, &v).0
        }
        1 => {
            // strip the placeholder colon: SIM(k, q) is not a :param
            let mut v = tokens.to_vec();
            let i = v.iter().position(|t| t == ":q").unwrap();
            v[i] = "q".into();
            join(rng, &v).0
        }
        2 => {
            // swap a keyword with its right neighbour
            let mut v = tokens.to_vec();
            let starts = [find("group").unwrap(), find("from").unwrap(), 0];
            let i = starts[rng.below(starts.len())];
            v.swap(i, i + 1);
            join(rng, &v).0
        }
        3 => {
            // inject junk at a random gap
            let mut v = tokens.to_vec();
            let pos = rng.below(v.len() + 1);
            v.insert(pos, JUNK[rng.below(JUNK.len())].to_string());
            join(rng, &v).0
        }
        4 => {
            // unterminated string literal at the end
            let (mut s, _) = join(rng, tokens);
            s.push_str(" 'never closed");
            s
        }
        5 => {
            // deep nesting: balanced parens around the eps argument
            // (rejected by num_of at small depth, by the parser's
            // recursion limit at large depth — never a stack overflow),
            // or a bare open-paren tower
            if rng.coin() {
                let k = 1 + rng.below(256);
                format!(
                    "SELECT g, SOFTAVG(v, SIM(k, :q), {}0.1{}) FROM t GROUP BY g",
                    "(".repeat(k),
                    ")".repeat(k)
                )
            } else {
                "(".repeat(64 + rng.below(4000))
            }
        }
        6 => {
            // truncation strictly before the GROUP BY column: no prefix
            // can carry a complete `GROUP BY <ident>` (all-ASCII input,
            // so every byte index is a char boundary)
            let (s, starts) = join(rng, tokens);
            let last = *starts.last().unwrap();
            let cut = 1 + rng.below(last - 1);
            s[..cut].to_string()
        }
        _ => {
            // split a required keyword in half
            let mut v = tokens.to_vec();
            let cands: Vec<usize> = ["select", "softavg", "group"]
                .iter()
                .filter_map(|t| find(t))
                .collect();
            let i = cands[rng.below(cands.len())];
            let t = v[i].clone();
            v[i] = format!("{} {}", &t[..3], &t[3..]);
            join(rng, &v).0
        }
    }
}

#[test]
fn fuzz_500_malformed_inputs_error_and_never_panic() {
    for seed in 0..500u64 {
        let mut rng = Rng::new(seed.wrapping_mul(0x517C_C1B7_2722_0A95).wrapping_add(1));
        let base = gen_valid(&mut rng, false);
        let sql = mutate(&mut rng, &base.tokens);

        let out = catch_unwind(AssertUnwindSafe(|| parse_query(&sql)))
            .unwrap_or_else(|_| panic!("seed {seed}: parse_query PANICKED on {sql:?}"));
        let err = match out {
            Err(e) => e,
            Ok(p) => panic!("seed {seed}: malformed input unexpectedly parsed to {p:?}:\n{sql:?}"),
        };
        assert!(
            !err.to_string().is_empty(),
            "seed {seed}: empty error message"
        );
    }
}

/// Stable errors: the same malformed input maps to the same error
/// string on every call (no address-dependent or iteration-order
/// noise in diagnostics).
#[test]
fn malformed_errors_are_deterministic() {
    for seed in 0..50u64 {
        let mut rng = Rng::new(seed.wrapping_mul(0x517C_C1B7_2722_0A95).wrapping_add(1));
        let base = gen_valid(&mut rng, false);
        let sql = mutate(&mut rng, &base.tokens);
        let e1 = parse_query(&sql).unwrap_err().to_string();
        let e2 = parse_query(&sql).unwrap_err().to_string();
        assert_eq!(e1, e2, "seed {seed}: unstable error for {sql:?}");
    }
}
