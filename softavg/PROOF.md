# Error bound of the removable softmax state: full derivation

This note gives the full derivation behind the lemma, the theorem and the corollary of the
paper's section "A Removable State with an Error Test", for `engines/core/src/lib.rs` and
`acc.rs` as released. It bounds what a first-order analysis drops: second-order terms, the
rounding of the bound's own accumulation, the use of computed instead of exact quantities inside
the charges, and subnormal results.

## 1. Model and assumptions

Notation: `u = 2^-53` (unit roundoff), `ω = 2^-1074` (smallest subnormal). The state is
`(μ, z, y, n, e_z, e_y, v_max, inexact)`; the code calls `y` `u`, `e_y` `err_u`, `v_max` `v_scale`.
Hats denote computed doubles. For the live rows `L` and the *stored* anchor `μ`,

    z* = Σ_{r in L} exp((s_r − μ)/ε),   y* = Σ_{r in L} exp((s_r − μ)/ε) v_r,   F* = y*/z*,

in real arithmetic over the given doubles `s_r, v_r, ε, μ` (fixed-input model: error in the scores
themselves, e.g. float32 inner products, is outside the statement).

- **A1** IEEE-754 binary64, round to nearest, no fused or contracted operations (Rust never
  contracts `a*b+c`; the core has no `mul_add`), no overflow.
  Consequences: `fl(a ± b) = (a ± b)(1+δ)`, `|δ| ≤ u`, exact when the result is subnormal;
  `fl(a·b) = ab(1+δ) + ν`, `|δ| ≤ u`, `|ν| ≤ ω/2`, same for division.
- **A2** `exp` returns a value within one ulp: `|exp_fl(t) − e^t| ≤ 2u e^t (1+2u) + ω`.
- **A3** `ε` finite and positive, scores and values finite, no NaN. (At `ε = 0`, `ε = ∞` and with
  `+∞` scores the state holds counts and plain value sums; see Section 6.)
- **A4** At most `N ≤ 2^26` operations (absorb, remove, merge; a merge counts the operations of
  both sides plus one) since the state was last *fresh* (empty, or rebuilt by a refold).
- Every `remove(s, v)` removes a row that was absorbed with the same `(s, v)` and is still live.

A state is *certified* when the sticky flag is clear; `check()` runs after every operation, so a
certified state passed `ẑ > ê_z` and `B̂ ≤ τ̂_n` after every operation since it was fresh.

## 2. Lemma (running bound, with explicit slack)

For a certified state,

    |ẑ − z*| ≤ (1 + η_N) ê_z + σ_N,      |ŷ − y*| ≤ (1 + η_N) ê_y + σ_N (1 + v_max),
    η_N = 2^-41 + 2^-46 N,               σ_N = 4 N^2 ω.

Proof, by induction over the operations.

**Absorb, s ≤ μ.** `x̂ = fl(fl(s−μ)/ε)` gives `θ := |x̂ − x| ≤ (2u+u²)|x| + ω/2`.
*Normal weight* (`ŵ ≥ 2^-1022`): then `|x| ≤ 709`, `θ ≤ 1.58e-13`, and
`|ŵ − w| ≤ w[θ(1+θ) + 2u(1+2u)(1+θ(1+θ))] ≤ w(2u + 2u|x|)(1+ξ)`, `ξ ≤ 1.6e-13`.
The code charges `2uŵ + 2u|x̂|ŵ` (computed): since `ŵ ≥ w(1 − 1.6e-13)`, `|x̂| ≥ |x|(1−2u)` and the
expression has at most six rounded operations, the computed charge is at least
`(1 − 3.3e-13)` times the needed amount.
*Subnormal or zero weight*: needed `ω + 2u e^{x̂} + wθ(1+θ)`, charged `ω + fl(2uŵ) + fl(2u|x̂|ŵ)`;
`2u e^{x̂} ≤ ω(1+…)` and each `fl` loses at most `ω/2`, so the deficiency is at most `2ω`
(for `|x| > 746`, `w < ω/4` and the same bound holds trivially).
*Product* `p̂ = fl(ŵ v)`: `|p̂ − wv| ≤ |ŵ−w||v| + u|ŵv| + ω/2`; charged
`3u|p̂| + 2u|x̂||p̂| + fl(t_z|v|) + [ω if p̂ subnormal]`: relative deficiency `≤ 3.4e-13`, absolute
deficiency `≤ 4ω + 2^-40 ω|v|`.
*Sum* `ẑ₁ = fl(ẑ₀ + ŵ)`: new rounding `≤ u|ẑ₁|(1+2u)`, charged `fl(u ẑ₁)` (exact up to `ω/2`).
*Accumulation* `ê ← fl(ê + c̃)`: one more factor `(1−u)`.
So if `E₀ ≤ (1+γ₀)ê₀ + σ₀`, then `E₁ ≤ (1+γ₁)ê₁ + σ₀ + 4ω(1+…)` with
`γ₁ ≤ max(γ₀, 3.5e-13) + 2u`.

**Remove.** Same computations with a subtraction; same rounding model; same charges. The anchor
does not move, so `z*`, `y*` stay sums at the stored `μ` whether or not the anchor row is live.

**Absorb, s > μ (anchor rises), and merge.** With `ρ = e^{(μ−s)/ε}`, `t = |x_s|`,
`|ρ̂ − ρ| ≤ ρ(2u + 2ut)(1+ξ)` (normal) or `≤ 2ω + ρ·rel` (subnormal), and

    |ẑ₁ − z*₁| ≤ E₀(ρ̂ + |ρ̂−ρ|) + ẑ₀|ρ̂−ρ| + uẑ₀ρ̂ + ω/2 + u ẑ₁(1+2u).

The terms in `ẑ₀` are covered by the charge `(3u + 2u|x̂_s|)ẑ₀ρ̂ + [ω ẑ₀]` as above (absolute
deficiency `≤ 2ωẑ₀ ≤ 4nω`). The carried error is multiplied by `ρ̂(1+a)` instead of `ρ̂`, with
`a ≤ 2u(1+t)(1+2ξ)`. This is the only place where the excess `γ` can grow by more than `O(u)`:
the new bound is `ê₁ ≥ (ê₀ρ̂ + c̃)(1−u)` with `c̃ ≥ u ẑ₁(1−u) ≥ u(1−u)` (because `ẑ₁ ≥ 1`), so
`γ₁ ≤ γ₀ + φ·a + 2u` with `φ = ê₀ρ̂/(ê₀ρ̂ + c̃) ≤ min(1, K e^{-t})`, `K = ê₀/u`. The state was
certified before the operation, hence `ê₀ < ẑ₀ ≤ 2n ≤ 2^28` and `K ≤ 2^81`;
`max_t min(1, K e^{-t})(1+t) = 1 + ln K ≤ 57.2`, so `φ a ≤ 115u` and `γ₁ ≤ γ₀ + 2^-46`.
A merge applies the same rescale to one side (the other scale is exactly 1) and adds both bounds;
`N = N_a + N_b + 1`.

Summing: `γ ≤ 2^-41 + 2^-46 N = η_N`; the absolute deficiencies sum to at most `4nω` per
operation, i.e. `σ_N = 4N²ω` (times `1 + v_max` for `y`). ∎

## 3. Theorem (delivered value)

Assume A1–A4. Let a state be certified, so `ẑ > ê_z` and `B̂ ≤ τ̂_n` with

    B = (ê_y + |F| ê_z)/(ẑ − ê_z) + u|F|,     τ_n = max(10^-10, 100 n u) · v_max,     F = fl(ŷ/ẑ).

If moreover **ẑ ≥ (1 + 2^-10) ê_z** (margin condition), then

    |F − F*| ≤ (1 + 2^-8) τ_n + 2^-900 (1 + v_max).

Proof. `ŷ/ẑ − F* = (δ_y − (ŷ/ẑ)δ_z)/z*`. With `D = ẑ − ê_z ≥ 2^-10 ê_z` and `η_N ≤ 2^-20(1+2^-21)`,
`z* ≥ ẑ − (1+η_N)ê_z − σ_N ≥ D(1 − 2^-10 − 2^-30)` (`ê_z ≥ u` for every state that ever held two
rows, so `σ_N/D ≤ 2^54 ω · 2^63 = 2^-957`). Hence
`|ŷ/ẑ − F*| ≤ (1+η_N)/(1 − 2^-10 − 2^-30) · (ê_y + fê_z)/D + 2σ_N(1+v_max)/D·…`,
the division adds `u f + ω/2`, the computed `B̂` and `τ̂_n` differ from their real values by at most
`8u` and `3u` relatively. Collecting: `|F − F*| ≤ (1+2^-9)(1+20u) τ_n + 2^-955(1+v_max)`. ∎

**What the released code tests.** `value_error_bound` returns ∞ unless `ẑ > ê_z`; it does not test
the margin, and it does not count `N`. Certification already implies the margin whenever
`|F| ≥ 2^-10 τ_n`, because `|F| ê_z / D ≤ B ≤ τ_n`. The uncovered case is therefore: delivered
value within `2^-10 τ_n` of zero **and** `ẑ/ê_z ∈ (1, 1 + 2^-10)`. A one-line change closes it: return ∞ unless `ẑ ≥ (1 + 2^-10) ê_z` (and set the flag when `N`
exceeds `2^26`).

## 4. Corollary (values of one sign)

If all values are positive (or all negative) and `min|v| ≥ 2^-16 max|v|`, every certified state
satisfies the margin condition, so the theorem applies to every delivered value. This covers the
rating workloads (IMDb 1–10, Amazon 1–5). It does not cover StackExchange answer scores or log
prices, which take both signs or zero.

Proof. Induction over operations; hypothesis: every earlier certified state had the margin, so by
the theorem `F̂ ≥ v' := v_min − 1.004 τ_max`, `τ_max ≤ 100·2^26·u·v_max < 7.5e-7 v_max`, hence
`ŷ ≥ v' ẑ(1−u)` at every earlier step. Term by term, every charge to `ê_y` except the current
`u|ŷ_t|` is at least `v'(1−2^-40)` times the matching charge to `ê_z`
(`3u|ŵv| ≥ v_min·2uŵ`, `2u|x̂||ŵv| ≥ v_min·2u|x̂|ŵ`, `(3u+2u|x̂_s|)|ŷ₀|ρ̂ ≥ v'(3u+2u|x̂_s|)ẑ₀ρ̂`,
`u|ŷ₁| ≥ v' u ẑ₁` at earlier steps). Write `ê_z = A_z + u ẑ_t`. Then `ê_y ≥ v' A_z (1−2^-40)`, and
certification gives `D ≥ ê_y/τ_n ≥ v' A_z/τ_n`. If `A_z ≥ ê_z/2`:
`D/ê_z ≥ v'/(2τ_n) ≥ 1.45e-5/(1.5e-6) > 9`. Otherwise `u ẑ_t > ê_z/2`, so
`D = ẑ_t − ê_z > ê_z(1/(2u) − 1)`. ∎

## 5. What the replay adds

The double-double replay (`experiments/bound`) checked 833.7M certified states against exact
values and found the true error at most 0.90 of `B`. That is evidence for the constants, not a
proof; the theorem is the proof, under A1–A4 and the margin condition.

## 6. Regimes outside A3

- `ε = ∞`: `z` is an exact count, `y` a plain sum; each add or subtract charges `u|ŷ|`.
- `ε = 0`, or a `+∞` score at finite `ε`: `z` counts the argmax set, `y` sums its values.
  Removing the last argmax row while rows remain sets the flag (the next anchor is unknown).
- Known gap: in the `+∞`-anchor regime at finite `ε`, **absorbing** a tie
  adds `v` to `y` without charging `u|ŷ|` (`remove` charges it). The delivered value is still a
  plain mean of at most `n` values, so its error is at most `n u v_max ≤ τ_n`; the guarantee
  holds, but `B` understates the error there. 
- `n = 0` after a removal resets the state to fresh; `Value` on an empty state is NULL.
- `v_max` is the largest `|v|` absorbed since the state was fresh, including rows that have left.
  The tolerance therefore scales with the historical maximum until the next refold.
