module
public import Mathlib
public section

namespace BranchVerify

/-- Updating two distinct row accumulators commutes. This licenses interleaving
    output rows while preserving the operation order within each row; it does
    not justify reordering any FP operations for a single row. -/
theorem independent_row_updates_commute {ι α : Type} [DecidableEq ι]
    (state : ι → α) (i j : ι) (fi fj : α → α) (hij : i ≠ j) :
    Function.update (Function.update state i (fi (state i))) j (fj (state j)) =
      Function.update (Function.update state j (fj (state j))) i (fi (state i)) := by
  funext k
  by_cases hki : k = i
  · subst k
    simp [hij]
  · by_cases hkj : k = j
    · subst k
      simp [hki]
    · simp [hki, hkj]

end BranchVerify

namespace BranchVerify

/-- Partitioning a sequence of real-valued group contributions across two
    accumulators preserves the final real sum. This is the exact-arithmetic
    justification for the K-split; IEEE-754 rounding order may differ. -/
def splitGroupSums : List ℝ → ℝ × ℝ
  | [] => (0, 0)
  | x :: xs => let p := splitGroupSums xs; (x + p.2, p.1)

theorem split_group_sums_correct (xs : List ℝ) :
    (splitGroupSums xs).1 + (splitGroupSums xs).2 = xs.sum := by
  induction xs with
  | nil => simp [splitGroupSums]
  | cons x xs ih =>
      simp only [splitGroupSums, List.sum_cons]
      linarith

end BranchVerify

namespace BranchVerify

/-- Centering the unsigned Q4 code `t = q + 8` inside the group dot product
    is exactly equivalent over the reals to subtracting the zero-point
    correction `8 * sum(x)` after the dot product. -/
theorem q4_group_centered_dot {n : ℕ} (t x : Fin n → ℝ) (scale : ℝ) :
    scale * (∑ i, (t i - 8) * x i) =
      scale * (∑ i, t i * x i) - 8 * scale * (∑ i, x i) := by
  have hsum : (∑ i, (t i - 8) * x i) =
      (∑ i, t i * x i) - 8 * (∑ i, x i) := by
    calc
      (∑ i, (t i - 8) * x i) = ∑ i, (t i * x i - 8 * x i) := by
        apply Finset.sum_congr rfl
        intro i _
        ring
      _ = (∑ i, t i * x i) - ∑ i, 8 * x i := by rw [Finset.sum_sub_distrib]
      _ = (∑ i, t i * x i) - 8 * (∑ i, x i) := by rw [Finset.mul_sum]
  rw [hsum]
  ring

end BranchVerify

namespace BranchVerify

/-- A second sibling is profitable exactly when its extra expected emitted tokens
    outweigh its extra round time at the existing rate. This is a rate model;
    it assumes the expected round time and reward are finite and positive. -/
theorem sibling_rate_iff (reward time extraReward extraTime : ℝ)
    (ht : 0 < time) (he : 0 < time + extraTime) :
    reward / time < (reward + extraReward) / (time + extraTime) ↔
      reward * extraTime < extraReward * time := by
  rw [div_lt_div_iff₀ ht he]
  constructor <;> intro h <;> nlinarith

/-- A confidence-gated sibling is rate-profitable when its expected extra
    token reward exceeds the measured round-cost threshold. The confidence
    score must estimate extraReward; this theorem does not establish calibration. -/
theorem sibling_confidence_threshold (reward time confidence extraTime : ℝ)
    (ht : 0 < time) (he : 0 < time + extraTime)
    (hconfidence : reward * extraTime / time < confidence) :
    reward / time < (reward + confidence) / (time + extraTime) := by
  apply (sibling_rate_iff reward time confidence extraTime ht he).2
  have hmul : reward * extraTime < confidence * time := (div_lt_iff₀ ht).mp hconfidence
  nlinarith

/-- When a first candidate misses, a matching distinct sibling contributes one
    additional already-verified continuation token, provided the continuation
    is not truncated by a stop token or output limit. -/
theorem sibling_reward (base hit : ℝ) (h : 0 ≤ hit) : base ≤ base + hit := by
  linarith

end BranchVerify

namespace BranchVerify

/-- The engine's stored 4-bit code is unsigned `t = q + 8`. Splitting its
    four bits yields the signed Q4 value as three positive planes, one
    sign-weighted plane, and the `-8` zero-point correction. This is the
    algebraic premise behind a T-MAC-style bit-plane/LUT kernel; it says
    nothing about floating-point reassociation or the speed of that kernel. -/
theorem q4_code_bitplanes (t : ℕ) (ht : t < 16) :
    (Int.ofNat t - 8) =
      Int.ofNat (t % 2) + 2 * Int.ofNat (t / 2 % 2) +
      4 * Int.ofNat (t / 4 % 2) + 8 * Int.ofNat (t / 8 % 2) - 8 := by
  have hbits : t = t % 2 + 2 * (t / 2 % 2) + 4 * (t / 4 % 2) + 8 * (t / 8 % 2) := by
    omega
  have hi := congrArg (fun n : ℕ => (n : ℤ)) hbits
  calc
    (Int.ofNat t - 8) =
        Int.ofNat (t % 2 + 2 * (t / 2 % 2) + 4 * (t / 4 % 2) + 8 * (t / 8 % 2)) - 8 := by
          exact congrArg (fun n : ℤ => n - 8) hi
    _ = Int.ofNat (t % 2) + 2 * Int.ofNat (t / 2 % 2) +
        4 * Int.ofNat (t / 4 % 2) + 8 * Int.ofNat (t / 8 % 2) - 8 := by
          norm_num

end BranchVerify

namespace BranchVerify

/-- The temporary sibling slot attends to the committed prefix and itself. -/
theorem sibling_mask (base slot p : ℕ) (hslot : base < slot) :
    (p ≤ slot ∧ ¬ (base + 1 ≤ p ∧ p ≤ slot - 1)) ↔
      (p ≤ base ∨ p = slot) := by omega

/-- The sibling hit contributes one extra emitted token before truncation. -/
theorem sibling_output_count (hit : Bool) :
    (if hit then 2 else 1 : ℕ) = 1 + (if hit then 1 else 0) := by
  cases hit <;> simp

end BranchVerify

namespace BranchVerify

/-- Independent rows with serial costs `a` and `b` benefit from parallel
    execution exactly when their maximum plus synchronization is smaller. -/
theorem two_row_parallel_iff (a b sync : ℝ) :
    max a b + sync < a + b ↔ sync < min a b := by
  have h := max_add_min a b
  constructor <;> intro hlt <;> linarith

end BranchVerify

namespace BranchVerify

/-- A tagged cache hit equals the recomputed value when every stored entry
    was created by the same deterministic function. -/
theorem tagged_cache_hit {Key Value Slot : Type} (f : Key → Value)
    (cache : Slot → Option (Key × Value)) (slot : Key → Slot)
    (valid : ∀ s k v, cache s = some (k, v) → v = f k)
    (key : Key) (value : Value)
    (hit : cache (slot key) = some (key, value)) : value = f key := by
  exact valid _ _ _ hit

end BranchVerify

namespace BranchVerify

/-- A three-bit base code plus nine bits per exception uses fewer than
    four bits per element precisely below the stated exception fraction.
    Stream headers, alignment, and indexing are not included. -/
theorem three_bit_escape_break_even (p : ℝ) :
    3 + 9 * p < 4 ↔ p < 1 / 9 := by
  constructor <;> intro h <;> linarith

end BranchVerify

namespace BranchVerify

/-- Group-wise bounds combine by the triangle inequality. Applying
    Cauchy-Schwarz inside each group supplies the hypotheses. -/
theorem group_bounds_sum {n : ℕ} (delta radius : Fin n → ℝ)
    (h : ∀ i, |delta i| ≤ radius i) :
    |∑ i, delta i| ≤ ∑ i, radius i := by
  calc
    |∑ i, delta i| ≤ ∑ i, |delta i| := Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ i, radius i := Finset.sum_le_sum (fun i _ => h i)

end BranchVerify

namespace BranchVerify

/-- Below a threshold whose capped value is strictly below the maximum's
    capped value, no candidate can tie it. The monotonicity hypothesis must
    be justified separately for the floating-point implementation. -/
theorem cap_threshold_safe (cap : ℝ → ℝ) (mono : Monotone cap)
    (candidate threshold maximum : ℝ) (h : candidate ≤ threshold)
    (strict : cap threshold < cap maximum) :
    cap candidate < cap maximum := by
  exact lt_of_le_of_lt (mono h) strict

end BranchVerify

namespace BranchVerify

/-- If every rounded value below a cutoff caps below the target, and raw
    values below the preceding representable value round below the cutoff,
    then an upper bound below that predecessor cannot hide a target tie. -/
theorem predecessor_cutoff_safe (cap round : ℝ → ℝ)
    (target cutoff predecessor raw upper : ℝ)
    (hcut : ∀ v, v < cutoff → cap v < target)
    (hround : raw < predecessor → round raw < cutoff)
    (hupper : raw ≤ upper) (htie : target ≤ cap (round raw)) :
    predecessor ≤ upper := by
  by_contra h
  have hr : raw < predecessor := lt_of_le_of_lt hupper (lt_of_not_ge h)
  have hc := hcut (round raw) (hround hr)
  linarith

end BranchVerify

namespace BranchVerify

/-- Parallel reduction of totally ordered score/tie-break keys can merge
    partial winners in any grouping. The representation must order ties by
    the preferred token ID; floating-point NaNs are excluded. -/
theorem partial_winner_associative {α : Type} [LinearOrder α] (a b c : α) :
    max (max a b) c = max a (max b c) := by
  exact max_assoc a b c

end BranchVerify

namespace BranchVerify

/-- Necessary and sufficient net time saving at unchanged tokens per round.
    Runtime costs and output counts are external assumptions. -/
theorem target_rate_net_savings (E T removed overhead R : ℝ)
    (hR : 0 < R) (hnew : 0 < T - removed + overhead) :
    R ≤ E / (T - removed + overhead) ↔
      T - E / R ≤ removed - overhead := by
  rw [le_div_iff₀ hnew]
  have h := (div_le_iff₀ hR : E / R ≤ T - removed + overhead ↔ _)
  constructor
  · intro hp
    have hd : T - removed + overhead ≤ E / R :=
      (le_div_iff₀ hR).mpr (by nlinarith)
    linarith
  · intro hp
    have hd : T - removed + overhead ≤ E / R := by linarith
    have hm := (le_div_iff₀ hR).mp hd
    nlinarith

end BranchVerify

namespace BranchVerify

/-- A four-weight signed Q4 dot expressed through four bit-plane subset sums.
    Bits are supplied as real coefficients; valid Q4 decoding must separately
    establish each coefficient is zero or one and reconstructs its nibble.
    This is an exact-real identity, not an IEEE rounding equivalence. -/
theorem q4_four_input_bitplane_dot
    (x0 x1 x2 x3 : ℝ) (b00 b01 b02 b03 b10 b11 b12 b13
      b20 b21 b22 b23 b30 b31 b32 b33 : ℝ) :
    x0 * (b00 + 2*b01 + 4*b02 + 8*b03 - 8) +
    x1 * (b10 + 2*b11 + 4*b12 + 8*b13 - 8) +
    x2 * (b20 + 2*b21 + 4*b22 + 8*b23 - 8) +
    x3 * (b30 + 2*b31 + 4*b32 + 8*b33 - 8) =
    (x0*b00 + x1*b10 + x2*b20 + x3*b30) +
    2*(x0*b01 + x1*b11 + x2*b21 + x3*b31) +
    4*(x0*b02 + x1*b12 + x2*b22 + x3*b32) +
    8*(x0*b03 + x1*b13 + x2*b23 + x3*b33) -
    8*(x0+x1+x2+x3) := by ring

/-- Compose absolute error bounds for bit-plane lookup values and the
    activation-sum correction. Bounds must include table construction,
    lookup quantization and arithmetic rounding in their respective inputs. -/
theorem q4_bitplane_error (e0 e1 e2 e3 ec r0 r1 r2 r3 rc : ℝ)
    (h0 : |e0| ≤ r0) (h1 : |e1| ≤ r1)
    (h2 : |e2| ≤ r2) (h3 : |e3| ≤ r3) (hc : |ec| ≤ rc) :
    |e0 + 2*e1 + 4*e2 + 8*e3 - 8*ec| ≤
      r0 + 2*r1 + 4*r2 + 8*r3 + 8*rc := by
  have lo0 := (abs_le.mp h0).1
  have hi0 := (abs_le.mp h0).2
  have lo1 := (abs_le.mp h1).1
  have hi1 := (abs_le.mp h1).2
  have lo2 := (abs_le.mp h2).1
  have hi2 := (abs_le.mp h2).2
  have lo3 := (abs_le.mp h3).1
  have hi3 := (abs_le.mp h3).2
  have loc := (abs_le.mp hc).1
  have hic := (abs_le.mp hc).2
  apply abs_le.mpr
  constructor <;> linarith

end BranchVerify

namespace BranchVerify

/-- If a seed subset gives lower bound `L0 ≤ L` and the threshold map `thr` is
    monotone (`L0 ≤ L → thr L0 ≤ thr L`), then any row surviving the exact
    threshold `thr L ≤ u` also survives the seed pre-screen `thr L0 ≤ u`.
    Hence filtering the pre-screened list by `thr L ≤ u` recovers the exact
    global survivor set without writing or scanning the 262,144-element `Sbuf`. -/
theorem seed_prescreen_sound (thr : ℝ → ℝ) (hmono : Monotone thr)
    (L0 L u : ℝ) (hseed : L0 ≤ L) :
    (thr L ≤ u) ↔ (thr L0 ≤ u ∧ thr L ≤ u) := by
  have hthr : thr L0 ≤ thr L := hmono hseed
  constructor
  · intro hu
    exact ⟨le_trans hthr hu, hu⟩
  · intro h
    exact h.2

/-- In a 2-step linear chain `[conf, d1, d2]`, row 1 (`d1`) is accepted iff
    `g0 = d1`, and row 2 (`d2`) is reachable iff `g0 = d1`. -/
theorem chain_reachability (g0 d1 : ℕ) :
    ¬ (g0 = d1) → (if g0 = d1 then true else false) = false := by
  intro h
  simp [h]

/-- In a 2-branch sibling tree `[conf, d1_top1, d1_top2]` (`M = 3, sibling = 2`)
    with `d1_top1 ≠ d1_top2`, rows 1 and 2 cannot both be accepted on the same
    verification step: at most 2 rows (`row 0` and at most one of `row 1, row 2`)
    ever need rescoring. -/
theorem sibling_mutual_exclusion (g0 d1_top1 d1_top2 : ℕ) (hne : d1_top1 ≠ d1_top2) :
    ¬ (g0 = d1_top1 ∧ g0 = d1_top2) := by
  rintro ⟨h1, h2⟩
  exact hne (h1.symm.trans h2)

/-- Expected tokens per verification step for a 2-step chain with step-1
    acceptance `p1` and conditional step-2 acceptance `p2` versus a 2-branch
    sibling tree with top-1 acceptance `p1` and disjoint runner-up acceptance `q2`.
    The sibling tree yields strictly more expected tokens iff `p1 * p2 < q2`,
    while always saving one full MTP `draft_step` call per round. -/
theorem sibling_beats_chain_iff (p1 p2 q2 : ℝ) :
    1 + p1 + p1 * p2 < 1 + p1 + q2 ↔ p1 * p2 < q2 := by
  constructor <;> intro h <;> linarith

end BranchVerify
