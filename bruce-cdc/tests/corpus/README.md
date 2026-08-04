# Golden pgoutput byte corpus (protocol version 1)

Hand-constructed replication-stream payloads per the pgoutput v1 wire
format (PostgreSQL docs "Logical Streaming Replication Protocol" /
"Streaming Replication Protocol"), consumed by `tests/conformance.rs`.
Regenerate (never edit by hand) with `python3 gen_corpus.py` in this
directory — the generator is deterministic, byte-for-byte.

## Framing

Each `.bin` file is a sequence of frames:

    u32 big-endian payload length || payload

One payload = one CopyData body exactly as the walsender sends it
(XLogData `'w'` envelope or primary keepalive `'k'`). The decoder under
test (`bruce_cdc::pgoutput::decode_wal`) consumes one payload per call.

## Files

| file | exercises |
|---|---|
| `multi_row_tx.bin` | one full transaction: Begin, Relation (replica identity `f`), 3 Inserts, 1 Delete with `'O'` old-tuple, Commit; every header field asserted exactly |
| `null_columns.bin` | SQL NULL columns (`'n'` marker) in Insert new-tuples and a `'K'` key-only Delete old-tuple |
| `text_edges.bin` | text values containing double/single quotes, embedded `\n` `\r\n` `\t`, SQL-injection-looking text, the empty string, 2/3-byte UTF-8 (`汉字 café ñ`) and 4-byte emoji (`😀🎬`) |
| `unchanged_toast.bin` | the `'u'` unchanged-TOAST column marker decoded as `TupleDatum::Unchanged`, DISTINCT from SQL NULL (`'n'` -> `TupleDatum::Null`); one Insert through the shared TupleData path plus an Update whose new tuple carries `'t'`/`'u'`/`'n'` in one tuple |
| `update_variants.bin` | one committed tx with the three Update old-tuple shapes (`'O'` full, `'K'` key-only, absent), then a REPLICA IDENTITY NOTHING relation whose Update the tx assembler rejects with a typed error naming `ALTER TABLE ... REPLICA IDENTITY` (the frames themselves decode fine) |
| `relation_added_column.bin` | Relation message re-sent mid-stream after ALTER TABLE ADD COLUMN, then a tuple carrying the new column — decoder must not panic; apply-plane contract: unmapped columns are ignored until re-snapshot |
| `ignored_messages.bin` | Origin (`'O'`), Type (`'Y'`), Truncate (`'T'`) — decoded as `Other(tag)`, never an error |
| `keepalive_reply.bin` | primary keepalive with reply-requested flag set and clear |
