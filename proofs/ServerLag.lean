module
public import Mathlib
public section

/-!
# Serving lag (engine server)

Model.  The engine commits tokens in bursts at times `a 0 ≤ a 1 ≤ …` (one burst per decode round).  The server is a
chain of FIFO stages (read, detokenise, format, write).  A stage that receives job `i` at time `a i` and needs `s i`
to handle it departs at `D i = max (D (i-1)) (a i) + s i`  (Lindley recursion).

Questions answered here:
* `stage_lag_le`        : if every job takes at most `S` and bursts are at least `S` apart, a stage adds at most `S` of
                          lag and never queues.  (So server lag = its own service time, whatever the run length.)
* `two_stage_lag_le`    : two stages in tandem add at most `S₁ + S₂` provided the bursts are `S₁ + S₂` apart.
* `holdback_safe`       : holding back `ℓ - 1` characters makes stop-string streaming retraction-free.
* `slices_concat`       : the streamed pieces concatenate to exactly the final text.
-/

namespace ServerLag

/-- Departure times of a FIFO stage: `D i = max (D (i-1)) (a i) + s i`, with `D (-1) = -∞` modelled by starting at `a 0`. -/
def depart (a s : ℕ → ℝ) : ℕ → ℝ
  | 0 => a 0 + s 0
  | n + 1 => max (depart a s n) (a (n + 1)) + s (n + 1)

/-- A stage never finishes a job before it arrives plus its service time. -/
theorem le_depart (a s : ℕ → ℝ) (n : ℕ) : a n + s n ≤ depart a s n := by
  cases n with
  | zero => simp [depart]
  | succ n => simp only [depart]; have := le_max_right (depart a s n) (a (n + 1)); linarith

/-- Departures are in order (FIFO), when service times are nonnegative. -/
theorem depart_mono (a s : ℕ → ℝ) (hs : ∀ i, 0 ≤ s i) (n : ℕ) : depart a s n ≤ depart a s (n + 1) := by
  simp only [depart]; have := le_max_left (depart a s n) (a (n + 1)); have := hs (n + 1); linarith

/-- **No queueing.**  If every job takes at most `S` and arrivals are at least `S` apart, then every job departs exactly
    its own service time after it arrives: the stage adds no waiting, so its lag is at most `S` for every `i`. -/
theorem stage_lag_eq (a s : ℕ → ℝ) (S : ℝ) (hs : ∀ i, s i ≤ S)
    (hgap : ∀ i, a i + S ≤ a (i + 1)) (n : ℕ) : depart a s n = a n + s n := by
  induction n with
  | zero => simp [depart]
  | succ n ih =>
    simp only [depart]; rw [ih]
    have : a n + s n ≤ a (n + 1) := by linarith [hs n, hgap n]
    rw [max_eq_right this]

theorem stage_lag_le (a s : ℕ → ℝ) (S : ℝ) (hs : ∀ i, s i ≤ S)
    (hgap : ∀ i, a i + S ≤ a (i + 1)) (n : ℕ) : depart a s n - a n ≤ S := by
  rw [stage_lag_eq a s S hs hgap n]; linarith [hs n]

/-- **Two stages in tandem.**  Stage 2 sees stage 1's departures.  With service bounds `S₁`, `S₂ ≥ 0` and engine bursts at
    least `S₁ + S₂` apart, the end-to-end lag is at most `S₁ + S₂`: no queue forms at either stage. -/
theorem two_stage_lag_le (a s₁ s₂ : ℕ → ℝ) (S₁ S₂ : ℝ) (hS₂ : 0 ≤ S₂) (h₁ : ∀ i, 0 ≤ s₁ i ∧ s₁ i ≤ S₁)
    (h₂ : ∀ i, s₂ i ≤ S₂) (hgap : ∀ i, a i + (S₁ + S₂) ≤ a (i + 1)) (n : ℕ) :
    depart (depart a s₁) s₂ n - a n ≤ S₁ + S₂ := by
  have e₁ : depart a s₁ = fun i => a i + s₁ i := funext fun i =>
    stage_lag_eq a s₁ (S₁ + S₂) (fun j => by linarith [(h₁ j).2]) hgap i
  rw [e₁]
  have hg : ∀ i, (a i + s₁ i) + S₂ ≤ a (i + 1) + s₁ (i + 1) := fun i => by
    linarith [(h₁ i).2, (h₁ (i + 1)).1, hgap i]
  have := stage_lag_le (fun i => a i + s₁ i) s₂ S₂ h₂ hg n
  linarith [(h₁ n).2]

/-! ### Streaming correctness -/

/-- **Hold-back safety.**  Let `T` be the text received so far and suppose the stop string `s` (length `ℓ`) does not occur
    inside `T`.  Whatever text `U` arrives next, any occurrence of `s` in `T ++ U` starts at an index `p` with
    `p + ℓ > T.length`, i.e. `p ≥ T.length - (ℓ - 1)`.  So the first `T.length - (ℓ - 1)` characters can be sent now and
    will never be cut off by a stop string. -/
theorem holdback_safe {α : Type} [DecidableEq α] (s T U : List α)
    (hnot : ∀ p, ¬ ((T.drop p).take s.length = s ∧ p + s.length ≤ T.length))
    (p : ℕ) (hocc : ((T ++ U).drop p).take s.length = s ∧ s ≠ []) :
    T.length < p + s.length := by
  by_contra hcon
  have hcon := not_lt.mp hcon
  apply hnot p
  refine ⟨?_, hcon⟩
  have hdrop : ((T ++ U).drop p).take s.length = (T.drop p).take s.length := by
    rw [List.drop_append_of_le_length (by omega : p ≤ T.length)]
    rw [List.take_append_of_le_length (by simp [List.length_drop]; omega)]
  rw [← hdrop]; exact hocc.1

theorem slices_take {α : Type} (T : List α) (b : ℕ → ℕ) (hmono : ∀ i, b i ≤ b (i + 1)) (h0 : b 0 = 0) (n : ℕ) :
    ((List.range n).map (fun i => (T.drop (b i)).take (b (i + 1) - b i))).flatten = T.take (b n) := by
  induction n with
  | zero => simp [h0]
  | succ n ih =>
    rw [List.range_succ, List.map_append, List.flatten_append, ih]
    simp only [List.map_cons, List.map_nil, List.flatten_cons, List.flatten_nil, List.append_nil]
    have hb : b (n + 1) = b n + (b (n + 1) - b n) := by have := hmono n; omega
    conv_rhs => rw [hb, List.take_add]

/-- **Pieces add up.**  If a text `T` is sent as the slices between boundaries `b 0 = 0 ≤ b 1 ≤ … ≤ b n = T.length`,
    the concatenation of the slices is `T`: streaming and non-streaming return the same text. -/
theorem slices_concat {α : Type} (T : List α) (b : ℕ → ℕ) (n : ℕ) (hmono : ∀ i, b i ≤ b (i + 1)) (h0 : b 0 = 0)
    (hn : b n = T.length) :
    ((List.range n).map (fun i => (T.drop (b i)).take (b (i + 1) - b i))).flatten = T := by
  rw [slices_take T b hmono h0 n, hn]; simp

end ServerLag
