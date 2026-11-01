module
public import Mathlib
public section

/-!
# Multi-slot prefix cache and idle parking (engine server)

* `reuse_sound`       : KV entries below the longest common prefix are valid for the new prompt (causality).
* `best_slot_max`     : the slot picked by longest common prefix reuses at least as much as any other slot.
* `evict_parks_active`: sacrificing slot `v` parks the active conversation there and leaves every other slot untouched.
* `park_competitive`  : a worker that spins for `τ` and then sleeps costs at most `(1 + τ/W)` times the best possible,
                        and never sleeps inside a request whose gaps are at most `τ`.
-/

namespace SlotCache

/-- Length of the longest common prefix of two token lists. -/
def lcp {α : Type} [DecidableEq α] : List α → List α → ℕ
  | a :: as, b :: bs => if a = b then lcp as bs + 1 else 0
  | _, _ => 0

theorem lcp_le_left {α : Type} [DecidableEq α] (a b : List α) : lcp a b ≤ a.length := by
  induction a generalizing b with
  | nil => simp [lcp]
  | cons x xs ih => cases b with
    | nil => simp [lcp]
    | cons y ys => simp only [lcp]; split_ifs <;> simp [ih ys]

theorem lcp_le_right {α : Type} [DecidableEq α] (a b : List α) : lcp a b ≤ b.length := by
  induction a generalizing b with
  | nil => simp [lcp]
  | cons x xs ih => cases b with
    | nil => simp [lcp]
    | cons y ys => simp only [lcp]; split_ifs <;> simp [ih ys]

/-- The two lists agree on their first `lcp` elements. -/
theorem take_lcp_eq {α : Type} [DecidableEq α] (a b : List α) : a.take (lcp a b) = b.take (lcp a b) := by
  induction a generalizing b with
  | nil => simp [lcp]
  | cons x xs ih => cases b with
    | nil => simp [lcp]
    | cons y ys =>
      simp only [lcp]; split_ifs with h
      · subst h; simp [ih ys]
      · simp

/-- **Reuse is sound.**  A causal model's cache entry at position `i` is a function `F` of the tokens up to `i`.
    If the stored conversation `keys` and the new `prompt` share `lcp` tokens, then every entry below `lcp` stored for
    `keys` is exactly the entry the new prompt would compute. -/
theorem reuse_sound {α β : Type} [DecidableEq α] (F : List α → β) (keys prompt : List α) (i : ℕ)
    (hi : i < lcp keys prompt) : F (keys.take (i + 1)) = F (prompt.take (i + 1)) := by
  have h := take_lcp_eq keys prompt
  have e1 : keys.take (i + 1) = (keys.take (lcp keys prompt)).take (i + 1) := by
    rw [List.take_take]; congr 1; omega
  have e2 : prompt.take (i + 1) = (prompt.take (lcp keys prompt)).take (i + 1) := by
    rw [List.take_take]; congr 1; omega
  rw [e1, e2, h]

/-- **Best slot.**  Choosing the slot with the longest common prefix reuses at least as much as any other slot. -/
theorem best_slot_max {α : Type} [DecidableEq α] (prompt : List α) (slots : List (List α)) (best : List α)
    (hbest : best ∈ slots.argmax (fun s => lcp s prompt)) :
    ∀ t ∈ slots, lcp t prompt ≤ lcp best prompt :=
  fun _ ht => List.le_of_mem_argmax ht hbest

/-- **Eviction parks the active conversation.**  Slots are indexed by `Option (Fin n)` with `none` the active one.
    Swapping the active slot with `v` puts the old active conversation at `v` and leaves every other slot as it was. -/
theorem evict_parks_active {n : ℕ} {S : Type} (st : Option (Fin n) → S) (v : Fin n) :
    (st ∘ Equiv.swap none (some v)) (some v) = st none ∧
    ∀ i : Fin n, i ≠ v → (st ∘ Equiv.swap none (some v)) (some i) = st (some i) := by
  refine ⟨by simp, fun i hi => ?_⟩
  simp only [Function.comp]
  rw [Equiv.swap_apply_of_ne_of_ne (by simp) (by simpa using hi)]

/-- Cost of an idle gap `x` for a worker that spins up to `τ` then sleeps (wake-up cost `W`). -/
noncomputable def park_cost (τ W x : ℝ) : ℝ := if x ≤ τ then x else τ + W
/-- Best possible cost with hindsight: spin the whole gap, or sleep at once, whichever is cheaper. -/
noncomputable def opt_cost (W x : ℝ) : ℝ := min x W

/-- **Competitive bound.**  For `0 < W ≤ τ`, the spin-then-park cost never exceeds `(1 + τ/W)` times the best possible. -/
theorem park_competitive (τ W x : ℝ) (hW : 0 < W) (hτ : W ≤ τ) (hx : 0 ≤ x) :
    park_cost τ W x ≤ (1 + τ / W) * opt_cost W x := by
  have hq : 1 ≤ τ / W := by rw [le_div_iff₀ hW]; linarith
  unfold park_cost opt_cost
  by_cases h1 : x ≤ τ
  · simp only [h1, ↓reduceIte]
    by_cases h2 : x ≤ W
    · rw [min_eq_left h2]; nlinarith
    · rw [min_eq_right (by linarith)]
      have : τ ≤ τ / W * W := by rw [div_mul_cancel₀ _ (ne_of_gt hW)]
      nlinarith
  · simp only [h1, ↓reduceIte]
    have hxW : W ≤ x := by linarith
    rw [min_eq_right hxW]
    have : τ ≤ τ / W * W := by rw [div_mul_cancel₀ _ (ne_of_gt hW)]
    nlinarith

/-- **No sleeping inside a request.**  If every gap inside a request is at most `τ`, a worker never reaches the sleep
    branch: the decode hot path is exactly the pure-spin path. -/
theorem no_park_in_request (τ W : ℝ) (gaps : List ℝ) (h : ∀ g ∈ gaps, g ≤ τ) :
    ∀ g ∈ gaps, park_cost τ W g = g := by
  intro g hg; unfold park_cost; simp only [h g hg, ↓reduceIte]

end SlotCache
