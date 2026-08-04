#!/usr/bin/env python3
"""Generate the golden pgoutput byte corpus for bruce-cdc/tests/conformance.rs.

Byte layout follows the pgoutput protocol version 1 wire format
(PostgreSQL docs: protocol-logicalrep-message-formats) plus the
replication-stream envelope (XLogData 'w' / keepalive 'k' from
protocol-replication).

Each .bin file is a sequence of frames; each frame is
  u32 big-endian payload length || payload
where payload is exactly one CopyData body as the walsender would send
it. The decoder under test consumes one payload per decode_wal() call.
"""
import struct, os

OUT = os.path.dirname(os.path.abspath(__file__))
os.makedirs(OUT, exist_ok=True)

def u8(v):   return struct.pack(">B", v)
def u16(v):  return struct.pack(">H", v)
def u32(v):  return struct.pack(">I", v)
def i32(v):  return struct.pack(">i", v)
def u64(v):  return struct.pack(">Q", v)
def i64(v):  return struct.pack(">q", v)
def cstr(s): return s.encode("utf-8") + b"\x00"

SEND_TS = 812345678901234  # arbitrary but fixed send timestamp (us since PG epoch)

def xlogdata(body, wal_start, wal_end):
    return b"w" + u64(wal_start) + u64(wal_end) + i64(SEND_TS) + body

def keepalive(wal_end, reply):
    return b"k" + u64(wal_end) + i64(SEND_TS) + u8(1 if reply else 0)

def begin(final_lsn, ts, xid):
    return b"B" + u64(final_lsn) + i64(ts) + u32(xid)

def commit(commit_lsn, end_lsn, ts, flags=0):
    return b"C" + u8(flags) + u64(commit_lsn) + u64(end_lsn) + i64(ts)

def relation(rel_id, ns, name, relident, cols):
    # cols: list of (flags, name, typoid, typmod)
    b = b"R" + u32(rel_id) + cstr(ns) + cstr(name) + relident + u16(len(cols))
    for flags, cname, typoid, typmod in cols:
        b += u8(flags) + cstr(cname) + u32(typoid) + i32(typmod)
    return b

def tuple_data(cols):
    # cols: list of None (SQL NULL, 'n'), "TOAST" sentinel ('u'),
    # or str (text 't', length-prefixed UTF-8 bytes)
    b = u16(len(cols))
    for c in cols:
        if c is None:
            b += b"n"
        elif c == ("TOAST",):
            b += b"u"
        else:
            enc = c.encode("utf-8")
            b += b"t" + i32(len(enc)) + enc
    return b

def insert(rel_id, cols):
    return b"I" + u32(rel_id) + b"N" + tuple_data(cols)

def delete(rel_id, cols, kind=b"O"):
    return b"D" + u32(rel_id) + kind + tuple_data(cols)

def update(rel_id, new, old=None, old_kind=b"O"):
    b = b"U" + u32(rel_id)
    if old is not None:
        b += old_kind + tuple_data(old)
    return b + b"N" + tuple_data(new)

def origin(lsn, name):
    return b"O" + u64(lsn) + cstr(name)

def type_msg(oid, ns, name):
    return b"Y" + u32(oid) + cstr(ns) + cstr(name)

def truncate(rel_ids, options=0):
    b = b"T" + u32(len(rel_ids)) + u8(options)
    for r in rel_ids:
        b += u32(r)
    return b

def write(name, payloads):
    path = os.path.join(OUT, name)
    with open(path, "wb") as f:
        for p in payloads:
            f.write(u32(len(p)) + p)
    print(f"{name}: {len(payloads)} frames, {os.path.getsize(path)} bytes")

TS = 812000000000000  # commit timestamp used inside pgoutput bodies
LSN = 0x0000000A_0000F000

# 1. multi_row_tx.bin — one committed transaction: Begin, Relation,
#    3 Inserts, 1 Delete (REPLICA IDENTITY FULL old tuple), Commit.
rel = relation(5001, "public", "corpus_t", b"f", [
    (1, "id", 23, -1),        # int4, part of key
    (0, "name", 25, -1),      # text
])
write("multi_row_tx.bin", [
    xlogdata(begin(LSN + 0x100, TS, 741), LSN, LSN + 0x200),
    xlogdata(rel, 0, LSN + 0x200),  # Relation carries wal_start 0 in real streams
    xlogdata(insert(5001, ["1", "alpha"]), LSN + 0x10, LSN + 0x200),
    xlogdata(insert(5001, ["2", "beta"]), LSN + 0x20, LSN + 0x200),
    xlogdata(insert(5001, ["3", "gamma"]), LSN + 0x30, LSN + 0x200),
    xlogdata(delete(5001, ["2", "beta"]), LSN + 0x40, LSN + 0x200),
    xlogdata(commit(LSN + 0x100, LSN + 0x108, TS), LSN + 0x100, LSN + 0x200),
])

# 2. null_columns.bin — Insert and Delete tuples carrying SQL NULLs ('n').
rel_n = relation(5002, "public", "nullable_t", b"f", [
    (1, "id", 23, -1), (0, "a", 25, -1), (0, "b", 701, -1),
])
write("null_columns.bin", [
    xlogdata(rel_n, 0, LSN),
    xlogdata(insert(5002, [None, "x", None]), LSN, LSN),
    xlogdata(insert(5002, ["7", None, "3.5"]), LSN, LSN),
    xlogdata(delete(5002, ["7", None, None], kind=b"K"), LSN, LSN),
])

# 3. text_edges.bin — text values with quotes, newlines, tabs, unicode
#    (2- and 3-byte sequences) and a 4-byte emoji; plus the empty string.
write("text_edges.bin", [
    xlogdata(insert(5003, ['he said "hi"', "line1\nline2\r\n\ttab"]), LSN, LSN),
    xlogdata(insert(5003, ["it's O'Brien; DROP TABLE--", ""]), LSN, LSN),
    xlogdata(insert(5003, ["汉字 café ñ", "emoji 😀🎬 end"]), LSN, LSN),
])

# 4. unchanged_toast.bin — the 'u' (unchanged TOAST datum) column
#    marker, kept DISTINCT from SQL NULL ('n'). Frame 1 drives 'u'
#    through the shared TupleData path (an Insert, as v0 did); frames
#    2-3 are the real habitat: an Update whose new tuple carries all
#    three markers ('t' / 'u' / 'n') in one tuple.
rel_toast = relation(5004, "public", "toast_t", b"d", [
    (1, "id", 23, -1), (0, "payload", 25, -1), (0, "note", 25, -1),
])
write("unchanged_toast.bin", [
    xlogdata(insert(5004, ["7", ("TOAST",), "z"]), LSN, LSN),
    xlogdata(rel_toast, 0, LSN),
    xlogdata(update(5004, ["7", ("TOAST",), None]), LSN, LSN),
])

# 4b. update_variants.bin — one committed transaction exercising the
#     three Update old-tuple shapes (REPLICA IDENTITY FULL 'O',
#     key-only 'K', absent), then a REPLICA IDENTITY NOTHING relation
#     whose Update the tx assembler must reject with a typed error
#     naming the ALTER TABLE fix (frames decode fine — assembly fails).
rel_u_full = relation(7001, "public", "upd_full_t", b"f", [
    (1, "id", 23, -1), (0, "val", 25, -1),
])
rel_u_dflt = relation(7002, "public", "upd_dflt_t", b"d", [
    (1, "id", 23, -1), (0, "val", 25, -1),
])
rel_u_none = relation(7003, "public", "upd_none_t", b"n", [
    (0, "id", 23, -1), (0, "val", 25, -1),
])
write("update_variants.bin", [
    xlogdata(begin(LSN + 0x100, TS, 933), LSN, LSN + 0x200),
    xlogdata(rel_u_full, 0, LSN + 0x200),
    xlogdata(update(7001, ["1", "new_v"], old=["1", "old_v"], old_kind=b"O"),
             LSN + 0x10, LSN + 0x200),
    xlogdata(rel_u_dflt, 0, LSN + 0x200),
    xlogdata(update(7002, ["9", "moved"], old=["2", None], old_kind=b"K"),
             LSN + 0x20, LSN + 0x200),
    xlogdata(update(7002, ["3", "direct"]), LSN + 0x30, LSN + 0x200),
    xlogdata(commit(LSN + 0x100, LSN + 0x108, TS), LSN + 0x100, LSN + 0x200),
    xlogdata(rel_u_none, 0, LSN + 0x200),
    xlogdata(update(7003, ["4", "x"]), LSN + 0x40, LSN + 0x200),
])

# 5. relation_added_column.bin — Relation re-sent mid-stream after
#    ALTER TABLE ADD COLUMN; subsequent tuples carry the extra column.
rel_v1 = relation(6001, "public", "grow_t", b"f", [
    (1, "id", 23, -1), (0, "name", 25, -1),
])
rel_v2 = relation(6001, "public", "grow_t", b"f", [
    (1, "id", 23, -1), (0, "name", 25, -1), (0, "extra", 25, -1),
])
write("relation_added_column.bin", [
    xlogdata(rel_v1, 0, LSN),
    xlogdata(insert(6001, ["1", "before"]), LSN, LSN),
    xlogdata(rel_v2, 0, LSN),
    xlogdata(insert(6001, ["2", "after", "surplus"]), LSN, LSN),
])

# 6. ignored_messages.bin — Origin ('O'), Type ('Y'), Truncate ('T'):
#    decoded as Other(tag), never an error, never applied.
write("ignored_messages.bin", [
    xlogdata(origin(LSN + 0x77, "replica_origin_1"), LSN, LSN),
    xlogdata(type_msg(16385, "public", "mood"), LSN, LSN),
    xlogdata(truncate([5001, 5002], options=1), LSN, LSN),
])

# 7. keepalive_reply.bin — primary keepalive frames, reply-requested
#    flag both set and clear.
write("keepalive_reply.bin", [
    keepalive(0x0000000B_00000010, True),
    keepalive(0x0000000B_00000020, False),
])
