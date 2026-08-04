//! Workstream 11 — pgoutput conformance against a golden byte corpus.
//!
//! The corpus lives in `tests/corpus/*.bin` (framing and inventory in
//! `tests/corpus/README.md`; regenerate with
//! `tests/corpus/gen_corpus.py`, byte-for-byte deterministic — each
//! frame is `u32 BE length || CopyData payload` per the pgoutput
//! protocol version 1 wire format from the PostgreSQL documentation).
//! Every file must decode without panic; field values are asserted
//! exactly. No PostgreSQL instance is needed here — this is the
//! decoder + apply-plane contract, byte level.

use std::path::PathBuf;

use bruce_cdc::apply::{Mirror, TableMap};
use bruce_cdc::pgoutput::{decode_wal, PgoutputMsg, TupleDatum, WalMsg};
use bruce_cdc::source::{CommittedTx, RowChange, TxAssembler};

// (The envelopes also carry a fixed send timestamp; the decoder
// discards it by design, so there is nothing to assert about it.)

/// Commit timestamp used inside corpus pgoutput bodies.
const TS: i64 = 812_000_000_000_000;
/// Base LSN used by the corpus.
const LSN: u64 = 0x0000_000A_0000_F000;

fn corpus(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/corpus")
        .join(name)
}

/// Split one corpus file into its `u32 BE length || payload` frames.
fn frames(name: &str) -> Vec<Vec<u8>> {
    let bytes = std::fs::read(corpus(name)).unwrap_or_else(|e| panic!("read {name}: {e}"));
    let mut out = Vec::new();
    let mut pos = 0;
    while pos < bytes.len() {
        let len = u32::from_be_bytes(bytes[pos..pos + 4].try_into().unwrap()) as usize;
        pos += 4;
        out.push(bytes[pos..pos + len].to_vec());
        pos += len;
    }
    out
}

/// Decode an XLogData frame, assert the envelope, return the message.
fn xlog(payload: &[u8], want_start: u64, want_end: u64) -> PgoutputMsg {
    match decode_wal(payload).expect("corpus frame must decode") {
        WalMsg::XLogData {
            wal_start,
            wal_end,
            msg,
        } => {
            assert_eq!(wal_start, want_start, "envelope wal_start");
            assert_eq!(wal_end, want_end, "envelope wal_end");
            msg
        }
        WalMsg::Keepalive { .. } => panic!("expected XLogData, got keepalive"),
    }
}

fn s(v: &str) -> TupleDatum {
    TupleDatum::text(v)
}

/// SQL NULL datum, for readable expected-value lists.
fn null() -> TupleDatum {
    TupleDatum::Null
}

#[test]
fn multi_row_tx_every_field_exact() {
    let f = frames("multi_row_tx.bin");
    assert_eq!(f.len(), 7);
    let end = LSN + 0x200;

    match xlog(&f[0], LSN, end) {
        PgoutputMsg::Begin {
            final_lsn,
            commit_ts_us,
            xid,
        } => {
            assert_eq!(final_lsn, LSN + 0x100);
            assert_eq!(commit_ts_us, TS);
            assert_eq!(xid, 741);
        }
        _ => panic!("frame 0: want Begin"),
    }
    match xlog(&f[1], 0, end) {
        PgoutputMsg::Relation {
            rel_id,
            namespace,
            name,
            replica_identity,
            columns,
        } => {
            assert_eq!(rel_id, 5001);
            assert_eq!(namespace, "public");
            assert_eq!(name, "corpus_t");
            assert_eq!(replica_identity, b'f');
            assert_eq!(columns, vec!["id".to_string(), "name".to_string()]);
        }
        _ => panic!("frame 1: want Relation"),
    }
    let want_rows = [["1", "alpha"], ["2", "beta"], ["3", "gamma"]];
    for (i, want) in want_rows.iter().enumerate() {
        match xlog(&f[2 + i], LSN + 0x10 * (i as u64 + 1), end) {
            PgoutputMsg::Insert { rel_id, new } => {
                assert_eq!(rel_id, 5001);
                assert_eq!(new, vec![s(want[0]), s(want[1])]);
            }
            _ => panic!("frame {}: want Insert", 2 + i),
        }
    }
    match xlog(&f[5], LSN + 0x40, end) {
        PgoutputMsg::Delete { rel_id, old } => {
            assert_eq!(rel_id, 5001);
            assert_eq!(
                old,
                vec![s("2"), s("beta")],
                "REPLICA IDENTITY FULL old tuple"
            );
        }
        _ => panic!("frame 5: want Delete"),
    }
    match xlog(&f[6], LSN + 0x100, end) {
        PgoutputMsg::Commit {
            end_lsn,
            commit_ts_us,
        } => {
            assert_eq!(end_lsn, LSN + 0x108, "ack point = commit end lsn");
            assert_eq!(commit_ts_us, TS);
        }
        _ => panic!("frame 6: want Commit"),
    }
}

#[test]
fn null_columns_decode_as_none() {
    let f = frames("null_columns.bin");
    assert_eq!(f.len(), 4);
    match xlog(&f[1], LSN, LSN) {
        PgoutputMsg::Insert { rel_id, new } => {
            assert_eq!(rel_id, 5002);
            assert_eq!(new, vec![null(), s("x"), null()]);
        }
        _ => panic!("want Insert"),
    }
    match xlog(&f[2], LSN, LSN) {
        PgoutputMsg::Insert { new, .. } => assert_eq!(new, vec![s("7"), null(), s("3.5")]),
        _ => panic!("want Insert"),
    }
    // key-only ('K') delete: non-key columns arrive as NULL
    match xlog(&f[3], LSN, LSN) {
        PgoutputMsg::Delete { rel_id, old } => {
            assert_eq!(rel_id, 5002);
            assert_eq!(old, vec![s("7"), null(), null()]);
        }
        _ => panic!("want Delete"),
    }
}

#[test]
fn text_edge_values_survive_byte_exact() {
    let f = frames("text_edges.bin");
    let want: [[&str; 2]; 3] = [
        ["he said \"hi\"", "line1\nline2\r\n\ttab"],
        ["it's O'Brien; DROP TABLE--", ""],
        ["汉字 café ñ", "emoji 😀🎬 end"],
    ];
    assert_eq!(f.len(), want.len());
    for (frame, w) in f.iter().zip(want.iter()) {
        match xlog(frame, LSN, LSN) {
            PgoutputMsg::Insert { rel_id, new } => {
                assert_eq!(rel_id, 5003);
                assert_eq!(new, vec![s(w[0]), s(w[1])]);
            }
            _ => panic!("want Insert"),
        }
    }
}

#[test]
fn unchanged_toast_is_distinct_from_null() {
    // DEFINED SEMANTICS (v1, supersedes the v0 pin): the 'u'
    // (unchanged TOAST) marker decodes as TupleDatum::Unchanged,
    // DISTINCT from SQL NULL ('n' -> TupleDatum::Null) — the
    // Update-capable apply path resolves Unchanged from the mirror's
    // current row, which demands the split. Documented in pgoutput.rs.
    let f = frames("unchanged_toast.bin");
    assert_eq!(f.len(), 3);
    match xlog(&f[0], LSN, LSN) {
        PgoutputMsg::Insert { rel_id, new } => {
            assert_eq!(rel_id, 5004);
            assert_eq!(new, vec![s("7"), TupleDatum::Unchanged, s("z")]);
        }
        _ => panic!("want Insert"),
    }
    // the real habitat: an Update new-tuple carrying all three
    // markers in one tuple — 't', 'u' and 'n' must stay 3-way distinct
    match xlog(&f[2], LSN, LSN) {
        PgoutputMsg::Update { rel_id, old, new } => {
            assert_eq!(rel_id, 5004);
            assert!(old.is_none(), "no old tuple in this frame");
            assert_eq!(new, vec![s("7"), TupleDatum::Unchanged, null()]);
            assert_ne!(new[1], new[2], "'u' and 'n' must not collapse");
        }
        _ => panic!("want Update"),
    }
}

#[test]
fn update_variants_assemble_into_row_changes() {
    // The three Update old-tuple shapes end-to-end through the
    // connection-free TxAssembler: 'O' (REPLICA IDENTITY FULL), 'K'
    // (DEFAULT, key changed), absent (DEFAULT, key untouched).
    let f = frames("update_variants.bin");
    assert_eq!(f.len(), 9);
    let mut asm = TxAssembler::new();
    let mut committed: Option<CommittedTx> = None;
    for frame in &f[..7] {
        match decode_wal(frame).unwrap() {
            WalMsg::XLogData { msg, .. } => {
                if let Some(tx) = asm.on_msg(msg).unwrap() {
                    committed = Some(tx);
                }
            }
            WalMsg::Keepalive { .. } => panic!("no keepalives in this corpus"),
        }
    }
    let tx = committed.expect("Commit must yield the transaction");
    assert_eq!(tx.end_lsn, LSN + 0x108);
    assert_eq!(tx.changes.len(), 3);
    match &tx.changes[0] {
        RowChange::Update { rel, old, new } => {
            assert_eq!(rel, "upd_full_t");
            let old = old.as_ref().expect("'O' old tuple present");
            assert_eq!(old[0], ("id".to_string(), s("1")));
            assert_eq!(old[1], ("val".to_string(), s("old_v")));
            assert_eq!(new[1], ("val".to_string(), s("new_v")));
        }
        _ => panic!("change 0: want Update"),
    }
    match &tx.changes[1] {
        RowChange::Update { rel, old, new } => {
            assert_eq!(rel, "upd_dflt_t");
            let old = old.as_ref().expect("'K' old tuple present");
            assert_eq!(old[0], ("id".to_string(), s("2")));
            assert_eq!(
                old[1],
                ("val".to_string(), null()),
                "key-only: non-key NULL"
            );
            assert_eq!(new[0], ("id".to_string(), s("9")), "pk change");
        }
        _ => panic!("change 1: want Update"),
    }
    match &tx.changes[2] {
        RowChange::Update { rel, old, new } => {
            assert_eq!(rel, "upd_dflt_t");
            assert!(old.is_none(), "key untouched: no old tuple");
            assert_eq!(new[1], ("val".to_string(), s("direct")));
        }
        _ => panic!("change 2: want Update"),
    }
}

#[test]
fn replica_identity_nothing_update_is_typed_error_naming_the_fix() {
    // DEFINED SEMANTICS: an Update for a relation with REPLICA
    // IDENTITY NOTHING cannot identify the old row; the assembler
    // rejects it with a typed error that names the ALTER TABLE fix.
    // (A real PG stream never carries this — PG refuses the UPDATE
    // statement itself with SQLSTATE 55000; pinned live in
    // tests/update_toast.rs — but a decoded stream is untrusted
    // input, so the guard must be total.)
    let f = frames("update_variants.bin");
    let mut asm = TxAssembler::new();
    let rel_msg = match decode_wal(&f[7]).unwrap() {
        WalMsg::XLogData { msg, .. } => msg,
        _ => panic!("want XLogData"),
    };
    assert!(matches!(
        rel_msg,
        PgoutputMsg::Relation {
            replica_identity: b'n',
            ..
        }
    ));
    asm.on_msg(rel_msg).unwrap();
    let upd_msg = match decode_wal(&f[8]).unwrap() {
        WalMsg::XLogData { msg, .. } => msg,
        _ => panic!("want XLogData"),
    };
    let err = asm.on_msg(upd_msg).unwrap_err().to_string();
    assert!(err.contains("REPLICA IDENTITY NOTHING"), "got: {err}");
    assert!(
        err.contains("ALTER TABLE upd_none_t REPLICA IDENTITY"),
        "the error must name the fix: {err}"
    );
}

#[test]
fn relation_resent_with_added_column_decodes() {
    let f = frames("relation_added_column.bin");
    assert_eq!(f.len(), 4);
    match xlog(&f[0], 0, LSN) {
        PgoutputMsg::Relation {
            rel_id, columns, ..
        } => {
            assert_eq!((rel_id, columns.len()), (6001, 2));
        }
        _ => panic!("want Relation v1"),
    }
    match xlog(&f[1], LSN, LSN) {
        PgoutputMsg::Insert { new, .. } => assert_eq!(new, vec![s("1"), s("before")]),
        _ => panic!("want Insert (2 cols)"),
    }
    // the mid-stream re-sent Relation after ALTER TABLE ADD COLUMN
    match xlog(&f[2], 0, LSN) {
        PgoutputMsg::Relation {
            rel_id,
            namespace,
            name,
            columns,
            ..
        } => {
            assert_eq!(rel_id, 6001);
            assert_eq!((namespace.as_str(), name.as_str()), ("public", "grow_t"));
            assert_eq!(
                columns,
                vec!["id".to_string(), "name".to_string(), "extra".to_string()]
            );
        }
        _ => panic!("want Relation v2"),
    }
    match xlog(&f[3], LSN, LSN) {
        PgoutputMsg::Insert { new, .. } => {
            assert_eq!(new, vec![s("2"), s("after"), s("surplus")]);
        }
        _ => panic!("want Insert (3 cols)"),
    }
}

#[test]
fn added_column_is_ignored_by_apply_until_resnapshot() {
    // DEFINED SEMANTICS (documented in source.rs): after ALTER TABLE
    // ADD COLUMN the re-sent Relation widens the tuples, but the
    // mirror's TableMap is fixed at snapshot time — columns not in the
    // map are ignored until a re-snapshot. A DROPPED mapped column
    // fails loudly instead ("tuple missing column").
    let names: Vec<String> = ["movie_id", "genre", "rating", "year", "e0", "e1"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    // snapshot rows travel the simple-query protocol: Option<String>
    let snap = |v: &str| Some(v.to_string());
    let seed = vec![vec![
        snap("1"),
        snap("action"),
        snap("5"),
        snap("2000"),
        snap("1"),
        snap("0"),
    ]];
    let mut m = Mirror::from_snapshot(TableMap::cdc_movies(), &names, &seed).unwrap();

    // widened tuple: one surplus column beyond the map
    let mut cols: Vec<(String, TupleDatum)> = names
        .iter()
        .cloned()
        .zip(vec![s("2"), s("drama"), s("7"), s("2001"), s("0"), s("1")])
        .collect();
    cols.push(("added_after_snapshot".to_string(), s("surplus")));
    let tx = CommittedTx {
        changes: vec![RowChange::Insert {
            rel: "cdc_movies".into(),
            cols,
        }],
        commit_ts_us: 0,
        end_lsn: 10,
    };
    assert_eq!(
        m.apply_tx(&tx).unwrap(),
        1,
        "surplus column must be ignored"
    );
    assert_eq!(m.n_rows(), 2);

    // narrowed tuple: a mapped column gone -> loud error, not drift
    let narrow: Vec<(String, TupleDatum)> = names[..5]
        .iter()
        .cloned()
        .zip(vec![s("3"), s("scifi"), s("8"), s("2002"), s("0.5")])
        .collect();
    let tx = CommittedTx {
        changes: vec![RowChange::Insert {
            rel: "cdc_movies".into(),
            cols: narrow,
        }],
        commit_ts_us: 0,
        end_lsn: 11,
    };
    let err = m.apply_tx(&tx).unwrap_err().to_string();
    assert!(err.contains("missing column"), "got: {err}");
}

#[test]
fn origin_type_truncate_are_ignored_not_errors() {
    let f = frames("ignored_messages.bin");
    let want_tags = [b'O', b'Y', b'T'];
    assert_eq!(f.len(), want_tags.len());
    for (frame, want) in f.iter().zip(want_tags.iter()) {
        match xlog(frame, LSN, LSN) {
            PgoutputMsg::Other(tag) => assert_eq!(tag, *want),
            _ => panic!("Origin/Type/Truncate must decode as Other"),
        }
    }
}

#[test]
fn keepalive_reply_flag_both_ways() {
    let f = frames("keepalive_reply.bin");
    assert_eq!(f.len(), 2);
    match decode_wal(&f[0]).unwrap() {
        WalMsg::Keepalive {
            wal_end,
            reply_requested,
        } => {
            assert_eq!(wal_end, 0x0000_000B_0000_0010);
            assert!(reply_requested, "reply-requested flag set");
        }
        _ => panic!("want Keepalive"),
    }
    match decode_wal(&f[1]).unwrap() {
        WalMsg::Keepalive {
            wal_end,
            reply_requested,
        } => {
            assert_eq!(wal_end, 0x0000_000B_0000_0020);
            assert!(!reply_requested, "reply-requested flag clear");
        }
        _ => panic!("want Keepalive"),
    }
}

#[test]
fn every_corpus_file_decodes_without_panic() {
    // Catch-all: any .bin dropped into the corpus dir is automatically
    // covered — every frame must decode Ok (panic = instant failure).
    let dir = corpus("");
    let mut seen = 0;
    for entry in std::fs::read_dir(&dir).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().and_then(|e| e.to_str()) != Some("bin") {
            continue;
        }
        seen += 1;
        let name = path.file_name().unwrap().to_string_lossy().into_owned();
        for (i, frame) in frames(&name).iter().enumerate() {
            decode_wal(frame).unwrap_or_else(|e| panic!("{name} frame {i} must decode: {e}"));
        }
    }
    assert_eq!(seen, 8, "corpus inventory drifted from README");
}
