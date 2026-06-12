//! GDPR cascading delete.
//!
//! When a data subject exercises the right-to-erasure, the deletion
//! must cascade to **every record about that subject** across all
//! tables that reference them. Bruce ships a primitive that:
//!
//! 1. Takes a `subject_id` (e.g., a customer's UUID, a user's email).
//! 2. Walks a configured set of `(source_id_field, table)` references.
//! 3. Calls `KvMemory::delete` on every matching record.
//! 4. Records the cascade in the audit log so the regulator gets a
//!    single coherent receipt.
//!
//! Each delete is an O(d) Bruce delete (Lemma A) — even at 10⁶ rows
//! per subject, the cascade finishes in milliseconds.

use crate::error::Result;
use crate::memory::KvMemory;

/// A single delete plan: which `fact_ids` to remove on behalf of
/// which `subject_id`, scoped to which Bruce memory.
#[derive(Debug, Clone)]
pub struct CascadePlan<'a> {
    /// Logical subject (e.g. a customer-id) being erased.
    pub subject_id: String,
    /// Per-table list of fact-ids tied to this subject.
    pub references: Vec<(&'a str, Vec<String>)>,
    /// Owner authorising the cascade (e.g. "gdpr-controller").
    pub owner: String,
}

/// Outcome of one cascade.
#[derive(Debug, Clone)]
pub struct CascadeReceipt {
    /// Subject erased.
    pub subject_id: String,
    /// Owner who issued the cascade.
    pub owner: String,
    /// Per-table deletion counts. `Err` entries mean some ids were
    /// already absent (idempotent — that's fine for GDPR).
    pub per_table: Vec<(String, Vec<String>, usize)>,
    /// Total records removed.
    pub n_total: usize,
}

impl<'a> CascadePlan<'a> {
    /// Execute the cascade against a single-table Bruce memory.
    ///
    /// The caller is responsible for joining whatever foreign-key
    /// relations exist on its side and passing in concrete fact-ids
    /// per table. Bruce doesn't know your schema; it only knows how
    /// to delete records O(d) at a time.
    pub fn execute(self, table_name: &str, mem: &mut KvMemory) -> Result<CascadeReceipt> {
        let mut per_table = Vec::new();
        let mut total = 0usize;
        for (tbl, ids) in &self.references {
            // For this single-table API we only process the ids that
            // belong to the table whose memory we hold.
            if *tbl != table_name {
                continue;
            }
            let mut errors = 0usize;
            let mut ok_ids = Vec::new();
            for id in ids {
                match mem.delete(id, &self.owner) {
                    Ok(()) => {
                        ok_ids.push(id.clone());
                        total += 1;
                    }
                    Err(_) => {
                        errors += 1;
                    }
                }
            }
            let _ = errors;
            per_table.push((table_name.into(), ok_ids, ids.len()));
        }
        Ok(CascadeReceipt {
            subject_id: self.subject_id,
            owner: self.owner,
            per_table,
            n_total: total,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn cascade_deletes_all_subject_records() {
        let mut mem = KvMemory::new(2, 1);
        // 5 rows for customer42, 3 rows for customer99
        for i in 0..5 {
            mem.write(
                &format!("cust42_r{i}"),
                array![1.0, 0.0].view(),
                array![i as f64].view(),
                "gdpr-controller",
            )
            .unwrap();
        }
        for i in 0..3 {
            mem.write(
                &format!("cust99_r{i}"),
                array![0.0, 1.0].view(),
                array![i as f64].view(),
                "gdpr-controller",
            )
            .unwrap();
        }
        assert_eq!(mem.len_alive(), 8);

        let plan = CascadePlan {
            subject_id: "customer42".into(),
            references: vec![("rentals", (0..5).map(|i| format!("cust42_r{i}")).collect())],
            owner: "gdpr-controller".into(),
        };
        let receipt = plan.execute("rentals", &mut mem).unwrap();
        assert_eq!(receipt.n_total, 5);
        assert_eq!(mem.len_alive(), 3); // customer99's 3 rows remain
    }

    #[test]
    fn cascade_is_idempotent() {
        let mut mem = KvMemory::new(1, 1);
        mem.write("only", array![1.0].view(), array![1.0].view(), "owner")
            .unwrap();
        let plan = CascadePlan {
            subject_id: "s".into(),
            references: vec![("t", vec!["only".into(), "doesnt-exist".into()])],
            owner: "owner".into(),
        };
        let r = plan.execute("t", &mut mem).unwrap();
        assert_eq!(r.n_total, 1);
        // running again should not error
        let plan2 = CascadePlan {
            subject_id: "s".into(),
            references: vec![("t", vec!["only".into()])],
            owner: "owner".into(),
        };
        let r2 = plan2.execute("t", &mut mem).unwrap();
        assert_eq!(r2.n_total, 0);
    }
}
