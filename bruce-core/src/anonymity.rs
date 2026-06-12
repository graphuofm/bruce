//! k-anonymity and l-diversity guards.
//!
//! Both are classical syntactic privacy definitions for tabular data
//! (Samarati 2001, Machanavajjhala 2006). Bruce uses them as **query
//! guards**: a hybrid query that returns fewer than `k` survivors, or
//! returns survivors whose sensitive attribute has fewer than `l`
//! distinct values, is rejected (or returned only with DP noise via
//! [`crate::dp`]).
//!
//! ### Why this matters for Bruce
//!
//! Without guards, a hybrid query with a very selective structural
//! filter (`author = X AND year = Y`) can identify a single record by
//! its semantic similarity to a query embedding. That defeats the
//! "retrieval memory" privacy story. k-anonymity says: refuse the
//! query if it pins down fewer than k records.

use ahash::AHashSet;

/// Privacy guard policy.
#[derive(Debug, Clone, Copy)]
pub struct AnonymityGuard {
    /// Reject if the survivor set has fewer than `k` records.
    pub k: usize,
    /// Reject if the survivors carry fewer than `l` distinct values
    /// of the sensitive attribute. `None` disables l-diversity.
    pub l: Option<usize>,
}

impl AnonymityGuard {
    /// k-anonymity only.
    pub fn k_anonymity(k: usize) -> Self {
        Self { k, l: None }
    }
    /// k-anonymity + l-diversity.
    pub fn k_and_l(k: usize, l: usize) -> Self {
        Self { k, l: Some(l) }
    }
}

/// Outcome of evaluating a guard.
#[derive(Debug, Clone)]
pub enum GuardOutcome {
    /// All checks passed; query may be answered exactly.
    Allow,
    /// Number of survivors is below `k`.
    DenyTooFewRecords {
        /// How many records actually survived.
        n: usize,
        /// Required k.
        k: usize,
    },
    /// Survivors have too few distinct sensitive values.
    DenyTooLowDiversity {
        /// How many distinct sensitive values were present.
        distinct: usize,
        /// Required l.
        l: usize,
    },
}

impl GuardOutcome {
    /// True iff `Allow`.
    pub fn allow(&self) -> bool {
        matches!(self, GuardOutcome::Allow)
    }
}

impl AnonymityGuard {
    /// Evaluate the guard against a list of survivors. `sensitives`
    /// supplies the sensitive-attribute value for each survivor (used
    /// only when l-diversity is enabled).
    pub fn evaluate<S>(&self, survivors: &[S]) -> GuardOutcome
    where
        S: Eq + std::hash::Hash,
    {
        let n = survivors.len();
        if n < self.k {
            return GuardOutcome::DenyTooFewRecords { n, k: self.k };
        }
        if let Some(l) = self.l {
            let distinct: AHashSet<&S> = survivors.iter().collect();
            if distinct.len() < l {
                return GuardOutcome::DenyTooLowDiversity {
                    distinct: distinct.len(),
                    l,
                };
            }
        }
        GuardOutcome::Allow
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn k_anonymity_blocks_singletons() {
        let g = AnonymityGuard::k_anonymity(5);
        let survivors = ["a", "b", "c"];
        let r = g.evaluate(&survivors);
        match r {
            GuardOutcome::DenyTooFewRecords { n: 3, k: 5 } => {}
            _ => panic!("expected DenyTooFewRecords"),
        }
    }

    #[test]
    fn k_anonymity_allows_large_enough() {
        let g = AnonymityGuard::k_anonymity(5);
        let survivors = ["a", "b", "c", "d", "e", "f"];
        assert!(g.evaluate(&survivors).allow());
    }

    #[test]
    fn l_diversity_blocks_homogeneous() {
        // 10 records but all sensitive=cancer → l-diversity fails for l=2
        let g = AnonymityGuard::k_and_l(5, 2);
        let survivors = vec!["cancer"; 10];
        let r = g.evaluate(&survivors);
        match r {
            GuardOutcome::DenyTooLowDiversity { distinct: 1, l: 2 } => {}
            _ => panic!("expected DenyTooLowDiversity"),
        }
    }

    #[test]
    fn l_diversity_allows_mixed() {
        let g = AnonymityGuard::k_and_l(3, 2);
        let survivors = vec!["cancer", "diabetes", "cancer", "asthma", "diabetes"];
        assert!(g.evaluate(&survivors).allow());
    }

    #[test]
    fn k_only_ignores_diversity() {
        let g = AnonymityGuard::k_anonymity(3);
        let survivors = vec!["cancer"; 5];
        assert!(g.evaluate(&survivors).allow()); // diversity not checked
    }
}
