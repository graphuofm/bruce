"""Tests for bruce.cascade_delete (GDPR cascade erasure)."""

from __future__ import annotations

import numpy as np

import bruce


class TestCascadeDelete:
    def test_deletes_all_subject_records(self):
        m = bruce.KvMemory(d_k=2, d_v=1)
        # 5 records for customer42, 3 for customer99
        for i in range(5):
            m.write(f"cust42_r{i}",
                     np.array([1.0, 0.0]),
                     np.array([float(i)]),
                     owner="dpo")
        for i in range(3):
            m.write(f"cust99_r{i}",
                     np.array([0.0, 1.0]),
                     np.array([float(i)]),
                     owner="dpo")
        assert m.len_alive == 8

        receipt = bruce.cascade_delete(
            m,
            subject_id="customer42",
            table_name="rentals",
            fact_ids=[f"cust42_r{i}" for i in range(5)],
            owner="dpo",
        )
        assert receipt["n_total"] == 5
        assert m.len_alive == 3            # customer99 records remain

    def test_receipt_records_per_table(self):
        m = bruce.KvMemory(d_k=1, d_v=1)
        m.write("a", np.array([0.0]), np.array([1.0]), owner="dpo")
        receipt = bruce.cascade_delete(
            m,
            subject_id="some-subject",
            table_name="orders",
            fact_ids=["a"],
            owner="dpo",
        )
        assert receipt["subject_id"] == "some-subject"
        assert receipt["owner"] == "dpo"
        assert len(receipt["per_table"]) == 1
        assert receipt["per_table"][0]["table"] == "orders"
        assert receipt["per_table"][0]["deleted_ids"] == ["a"]

    def test_idempotent_on_missing_ids(self):
        m = bruce.KvMemory(d_k=1, d_v=1)
        m.write("real", np.array([0.0]), np.array([1.0]), owner="dpo")
        # request deletion of 'real' AND 'doesnt-exist'; only 'real'
        # is actually deleted, no exception
        receipt = bruce.cascade_delete(
            m,
            subject_id="s",
            table_name="t",
            fact_ids=["real", "doesnt-exist"],
            owner="dpo",
        )
        assert receipt["n_total"] == 1

    def test_owner_must_match(self):
        m = bruce.KvMemory(d_k=1, d_v=1)
        m.write("x", np.array([0.0]), np.array([1.0]), owner="alice")
        receipt = bruce.cascade_delete(
            m,
            subject_id="s",
            table_name="t",
            fact_ids=["x"],
            owner="mallory",
        )
        # mallory wasn't the owner so nothing was deleted
        assert receipt["n_total"] == 0
        assert m.len_alive == 1
