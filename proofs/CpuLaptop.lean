module
public import Mathlib
public section

/-! CPU/laptop conjectures for Gemma-4-E2B QAT-int4 (Ryzen 7 7730U).
    Each theorem is the arithmetic/logic core of a conjecture; premises that come from
    the checkpoint, vLLM source or hardware are named in the docstring. -/

namespace CpuLaptop

/-- K1a. Certified head screen is sound over the reals: if `i` is a true argmax and every
    score is within its radius, `i` is never discarded by the interval test. -/
theorem screen_keeps_argmax {n : ℕ} (t s ρ : Fin n → ℝ) (i : Fin n)
    (hb : ∀ j, |t j - s j| ≤ ρ j) (hi : ∀ j, t j ≤ t i) :
    ∀ j, s j - ρ j ≤ s i + ρ i := by
  intro j
  have h1 := (abs_le.mp (hb j)).1
  have h2 := (abs_le.mp (hb i)).2
  have h3 := hi j
  linarith

/-- K1b. Byte accounting for the head: BF16 head = 805,306,368 B; int8 screen table with a
    4-byte scale per row plus `r` BF16 survivor rows. Positive saving iff r ≤ 130,730. -/
theorem head_saving_iff (r : ℕ) :
    (262144 * 1536 + 262144 * 4 + r * 3072 < 805306368) ↔ r ≤ 130730 := by
  omega

/-- K2. Per-token KV bytes of the KV-owning layers (12 sliding of head dim 256, 3 full of
    head dim 512, 1 KV head, K and V, BF16) is 18,432, and with the window cap the context
    cost at 32,768 tokens is 207,618,048 B. Premise: sliding layers are really capped. -/
def kvBytes (n : ℕ) : ℕ := 12 * 2 * 256 * 2 * min n 512 + 3 * 2 * 512 * 2 * n
theorem kv_at_32k : kvBytes 32768 = 207618048 := by decide +kernel
theorem kv_per_token_long (n : ℕ) (h : 512 ≤ n) :
    kvBytes n = 12288 * 512 + 6144 * n := by
  unfold kvBytes; rw [min_eq_right h]

/-- K3a. Causal look-ahead composes additively. `f` has look-ahead `a` when output `t`
    depends only on inputs `≤ t + a`. -/
def Lookahead (f : (ℕ → ℝ) → ℕ → ℝ) (a : ℕ) : Prop :=
  ∀ x y t, (∀ s, s ≤ t + a → x s = y s) → f x t = f y t

theorem lookahead_comp (f g : (ℕ → ℝ) → ℕ → ℝ) (a b : ℕ)
    (hf : Lookahead f a) (hg : Lookahead g b) : Lookahead (fun x => g (f x)) (a + b) := by
  intro x y t h
  apply hg
  intro s hs
  apply hf
  intro u hu
  exact h u (by omega)

/-- K3b. Streaming prefix stability: truncating/altering the input after `T` leaves
    outputs `t ≤ T - a` unchanged. -/
theorem prefix_stable (f : (ℕ → ℝ) → ℕ → ℝ) (a : ℕ) (hf : Lookahead f a)
    (x y : ℕ → ℝ) (T t : ℕ) (hxy : ∀ s, s ≤ T → x s = y s) (ht : t + a ≤ T) :
    f x t = f y t := by
  apply hf
  intro s hs
  exact hxy s (by omega)

/-- K4. Speculative decoding can only pay if acceptance exceeds the extra per-draft cost:
    expected extra tokens `∑_{i=1..k} α^i ≤ k·α`, so gain `> k·c` needs `α > c`. -/
theorem spec_needs_alpha_gt_c (α c : ℝ) (k : ℕ) (hk : 0 < k) (h0 : 0 ≤ α) (h1 : α ≤ 1)
    (hpay : (k : ℝ) * c < ∑ i ∈ Finset.range k, α ^ (i + 1)) : c < α := by
  have hle : ∑ i ∈ Finset.range k, α ^ (i + 1) ≤ k * α := by
    calc ∑ i ∈ Finset.range k, α ^ (i + 1) ≤ ∑ _i ∈ Finset.range k, α :=
          Finset.sum_le_sum fun i _ => pow_le_of_le_one h0 h1 (by omega)
      _ = k * α := by simp
  have hk' : (0 : ℝ) < k := by exact_mod_cast hk
  nlinarith

/-- K5. PLE table is 4,697,620,480 B; one token touches 17,920 B of it, so a mapped table
    has resident bytes ≤ 17,920·T for T distinct tokens. -/
theorem ple_table_bytes : 262144 * 35 * 256 * 2 = 4697620480 := by norm_num
theorem ple_resident (T : ℕ) (hT : T < 262144) : 17920 * T < 262144 * 35 * 256 * 2 := by
  omega



/-- K3c. Strided look-ahead. `SLook f s a`: output `t` depends only on inputs `≤ s*t + a`.
    Composition `g ∘ f` has stride `s1*s2` and look-ahead `a1 + s1*a2`. The audio
    sub-sampler is two 3x3 stride-2 pad-1 convs (s=2,a=1 each) so total a = 1 + 2*1 = 3
    mel frames, stride 4. -/
def SLook (f : (ℕ → ℝ) → ℕ → ℝ) (s a : ℕ) : Prop :=
  ∀ x y t, (∀ i, i ≤ s * t + a → x i = y i) → f x t = f y t

theorem slook_comp (f g : (ℕ → ℝ) → ℕ → ℝ) (s1 a1 s2 a2 : ℕ)
    (hf : SLook f s1 a1) (hg : SLook g s2 a2) :
    SLook (fun x => g (f x)) (s1 * s2) (a1 + s1 * a2) := by
  intro x y t h
  apply hg
  intro u hu
  apply hf
  intro i hi
  apply h
  have : s1 * u ≤ s1 * (s2 * t + a2) := Nat.mul_le_mul_left _ hu
  nlinarith

theorem audio_subsampler_lookahead : (1 + 2 * 1 = 3) ∧ (2 * 2 = 4) := by decide

/-- K6. Drafter economics in the bandwidth-bound regime with one draft token.
    One round reads `Bt + Bd` bytes and yields `1 + α` expected tokens; plain decode reads
    `Bt` per token. The round is faster iff `α·Bt > Bd`. -/
theorem spec_k1_iff (α Bt Bd : ℝ) (hα : 0 ≤ α) (_hBt : 0 < Bt) :
    (Bt + Bd) / (1 + α) < Bt ↔ Bd < α * Bt := by
  have h : (0:ℝ) < 1 + α := by linarith
  rw [div_lt_iff₀ h]
  constructor <;> intro k <;> nlinarith

/-- K7. Byte budget per generated token (MB). Ollama gemma4:e2b (read from its GGUF:
    Q4_K 748 + Q6_K 453 blocks + Q6_K head 330) reads 1531 MB per token. Stock vLLM on the QAT
    checkpoint reads 1056 (int4 body incl. scales) + 805 (BF16 head) = 1861 MB, i.e. 330 MB
    more, so plain decoding loses on a bandwidth-bound machine. -/
theorem ollama_bytes : (748 + 453 + 330 : ℕ) = 1531 := by norm_num
theorem stock_bytes : (1056 + 805 : ℕ) = 1861 ∧ 1861 - 1531 = 330 := by norm_num

/-- K8. With the 158 MB shipped drafter and k=3 drafts, one verify round reads
    1861 + 3*158 MB and yields E = 1+α+α²+α³ expected tokens. If α ≥ 1/2 the bytes per token
    are below Ollama's 1531 MB even with the stock BF16 head. -/
theorem spec_beats_ollama_bytes (α : ℝ) (h : (1:ℝ)/2 ≤ α) :
    (1861 + 3 * 158) / (1 + α + α^2 + α^3) < 1531 := by
  have hpos : (0:ℝ) < 1 + α + α^2 + α^3 := by positivity
  rw [div_lt_iff₀ hpos]
  nlinarith [sq_nonneg α, mul_pos (by linarith : (0:ℝ) < α) (by linarith : (0:ℝ) < α)]

/-- K5b. RAM budget. Stock vLLM allocates the PLE table as an anonymous parameter, so the
    resident model is the whole 8,315,979,014 B checkpoint; mapping the table leaves
    3,618,358,534 B anonymous, which fits the ~6.3 GB available (measured with a normal desktop session open). -/
theorem anon_bytes_after_ple_map :
    8315979014 - 4697620480 = 3618358534 ∧ 3618358534 < 6300000000 ∧ 6300000000 < 8315979014 := by
  norm_num

/-- K1c. Cauchy–Schwarz radius for the weight-quantisation error of a head row:
    (∑ e_i h_i)² ≤ (∑ e_i²)(∑ h_i²), i.e. |⟨w - ŵ, h⟩| ≤ ‖w - ŵ‖₂‖h‖₂. -/
theorem l2_radius_sq {n : ℕ} (e h : Fin n → ℝ) :
    (∑ i, e i * h i) ^ 2 ≤ (∑ i, e i ^ 2) * (∑ i, h i ^ 2) :=
  Finset.sum_mul_sq_le_sq_mul_sq Finset.univ e h

/-- K1d. Activation-rounding radius: if every coordinate of `h` is rounded by at most δ,
    the score changes by at most δ·‖w‖₁. -/
theorem act_quant_radius {n : ℕ} (w h h' : Fin n → ℝ) (δ : ℝ)
    (hb : ∀ i, |h i - h' i| ≤ δ) :
    |∑ i, w i * (h i - h' i)| ≤ δ * ∑ i, |w i| := by
  calc |∑ i, w i * (h i - h' i)| ≤ ∑ i, |w i * (h i - h' i)| := Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ i, |w i| * δ := Finset.sum_le_sum fun i _ => by
        rw [abs_mul]; exact mul_le_mul_of_nonneg_left (hb i) (abs_nonneg _)
    _ = δ * ∑ i, |w i| := by rw [← Finset.sum_mul, mul_comm]

/-- K9. Body time with per-call overhead: ∑ (b_i/BW + o) = (∑ b_i)/BW + n·o. -/
theorem body_time (n : ℕ) (b : Fin n → ℝ) (BW o : ℝ) :
    ∑ i, (b i / BW + o) = (∑ i, b i) / BW + n * o := by
  rw [Finset.sum_add_distrib, Finset.sum_div]; simp

/-- K10. Speculative round time: k draft steps plus a verify pass that is memory-bound
    (Tm) or compute-bound ((k+1)·Tc). When compute fits under the memory time the round is
    flat in the number of verified positions. -/
theorem spec_round_flat (k : ℕ) (td Tm Tc : ℝ) (h : ((k : ℝ) + 1) * Tc ≤ Tm) :
    (k : ℝ) * td + max Tm (((k : ℝ) + 1) * Tc) = Tm + k * td := by
  rw [max_eq_left h]; ring

/-- K11a. A BF16 word is exactly sign·2¹⁵ + exp·2⁷ + mant, and the three fields are recovered
    uniquely. So storing (sign, mant) in one byte and a small exponent *index* is lossless
    whenever the index maps back to the exponent. -/
theorem bf16_fields_unique (s e m : ℕ) (hs : s < 2) (he : e < 256) (hm : m < 128) :
    (s * 32768 + e * 128 + m) / 32768 = s ∧ ((s * 32768 + e * 128 + m) / 128) % 256 = e ∧
      (s * 32768 + e * 128 + m) % 128 = m := by
  omega

/-- K11b. Codebook round trip: if every exponent present is some `tab i`, the index chosen for it
    decodes back to it. -/
theorem codebook_roundtrip (tab : Fin 8 → ℕ) (e : ℕ) (h : ∃ i, tab i = e) :
    tab (Classical.choose h) = e := Classical.choose_spec h

/-- K11c. Head bytes: 262144·1536 weights at 11 bits = 553,648,128 B, saving exactly
    251,658,240 B (5/16 of 805,306,368 B) per decoded token, with an exact correction list because the
    measured head needs an exception list: see patch_exact (1094 weights outside the top-8 exponents). -/
theorem packed_head_bytes :
    262144 * 1536 * 11 / 8 = 553648128 ∧ 805306368 - 553648128 = 251658240 ∧
    251658240 * 16 = 5 * 805306368 := by
  norm_num

/-- K12. The 1.4x budget, checked rather than asserted. Bytes per round (MB): int4 body 1056,
    drafter 75, head h. At 21.5 GB/s a round takes (1131 + h)/21.5 ms. With E = 2 tokens per
    round, throughput is 2000·21.5/(1131 + h) tok/s. It reaches 31 tok/s iff h ≤ 43000/31 − 1131 ≈ 256.1 MB,
    so the exact 554 MB packed head gives at most 25.5 tok/s and cannot reach 31. -/
theorem budget_head_cap (h : ℝ) (hh : 0 ≤ h) :
    31 ≤ 2000 * 21.5 / (1131 + h) ↔ h ≤ 43000 / 31 - 1131 := by
  have hp : (0:ℝ) < 1131 + h := by linarith
  rw [le_div_iff₀ hp]
  constructor <;> intro k <;> linarith

theorem packed_head_throughput_lt_31 : 2000 * 21.5 / (1131 + 554 : ℝ) < 31 := by
  rw [div_lt_iff₀ (by norm_num)]; norm_num

/-- K11d. Exact correction list. If the decoded weights `d` agree with the true weights `w`
    outside a finite exception set `E`, then adding `∑_{i∈E} (w_i - d_i)·x_i` to the decoded dot
    product gives the true dot product. For the real head, |E| = 1094 of 402,653,184 weights
    (exponents 125 and 126 do not fit an 8-entry codebook). -/
theorem patch_exact {n : ℕ} (w d x : Fin n → ℝ) (E : Finset (Fin n))
    (h : ∀ i, i ∉ E → d i = w i) :
    ∑ i, w i * x i = ∑ i, d i * x i + ∑ i ∈ E, (w i - d i) * x i := by
  have hz : ∀ i ∈ (Finset.univ : Finset (Fin n)), i ∉ E → (w i - d i) * x i = 0 := by
    intro i _ hi; rw [h i hi]; ring
  have hsub : ∑ i ∈ E, (w i - d i) * x i = ∑ i, (w i - d i) * x i :=
    Finset.sum_subset (Finset.subset_univ E) hz
  rw [hsub, ← Finset.sum_add_distrib]
  apply Finset.sum_congr rfl; intro i _; ring

/-- K13a. W4A8 kernel safety. A weight nibble q ∈ [0,15] times an int8 activation |a| ≤ 127: one
    vpmaddubsw pair sum is ≤ 2·15·127 = 3810 (no saturation); the lo+hi nibble sum for a chunk is
    ≤ 7620; four chunks (one group of 32) accumulate to ≤ 30480 < 32767, so the 16-bit accumulator
    cannot overflow before the single vpmaddwd widening. -/
theorem w4a8_no_overflow :
    2 * 15 * 127 = 3810 ∧ 2 * (2 * 15 * 127) = 7620 ∧ 4 * (2 * (2 * 15 * 127)) = 30480 ∧ 30480 ≤ 32767 := by
  norm_num

/-- K13b. Offset identity used by the kernel: ∑(q-8)·a = ∑ q·a − 8·∑ a. -/
theorem w4a8_offset_identity {n : ℕ} (q a : Fin n → ℝ) :
    ∑ i, (q i - 8) * a i = ∑ i, q i * a i - 8 * ∑ i, a i := by
  simp only [sub_mul, Finset.sum_sub_distrib, Finset.mul_sum]

/-- K13c. Activation-rounding error of the int8 path, per group of n values: if every activation is
    rounded by at most δ (δ = scale/2) and the weight nibbles satisfy 0 ≤ q ≤ 15, then
    |∑ (q_i − 8)(x_i − x̂_i)| ≤ 8·n·δ. With n = 32 and δ = s_x/2 this is 128·s_x per group. -/
theorem w4a8_group_error {n : ℕ} (q x xh : Fin n → ℝ) (δ : ℝ)
    (hq0 : ∀ i, 0 ≤ q i) (hq1 : ∀ i, q i ≤ 15) (hx : ∀ i, |x i - xh i| ≤ δ) :
    |∑ i, (q i - 8) * (x i - xh i)| ≤ 8 * n * δ := by
  have hq : ∀ i, |q i - 8| ≤ 8 := fun i => by rw [abs_le]; constructor <;> linarith [hq0 i, hq1 i]
  calc |∑ i, (q i - 8) * (x i - xh i)| ≤ ∑ i, |(q i - 8) * (x i - xh i)| := Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ _i : Fin n, 8 * δ := Finset.sum_le_sum fun i _ => by
        rw [abs_mul]; exact mul_le_mul (hq i) (hx i) (abs_nonneg _) (by norm_num)
    _ = 8 * n * δ := by simp; ring

/-- K13d. Weighted activation quantisation bound. For known signed group
    weights, the generic 8*n bound can be tightened to the group's L1 weight
    norm: |Σ qᵢ(xᵢ−x̂ᵢ)| ≤ δ Σ|qᵢ|. -/
theorem w4a8_weighted_group_error {n : ℕ} (q x xh : Fin n → ℝ) (δ : ℝ)
    (hx : ∀ i, |x i - xh i| ≤ δ) :
    |∑ i, q i * (x i - xh i)| ≤ δ * ∑ i, |q i| := by
  calc
    |∑ i, q i * (x i - xh i)| ≤ ∑ i, |q i * (x i - xh i)| :=
      Finset.abs_sum_le_sum_abs _ _
    _ = ∑ i, |q i| * |x i - xh i| := by
      apply Finset.sum_congr rfl
      intro i _
      rw [abs_mul]
    _ ≤ ∑ i, |q i| * δ := Finset.sum_le_sum fun i _ =>
      mul_le_mul_of_nonneg_left (hx i) (abs_nonneg (q i))
    _ = δ * ∑ i, |q i| := by
      rw [Finset.mul_sum]
      apply Finset.sum_congr rfl
      intro i _
      ring

/-- K13e. Residual-aware variant of K13d: if the exact per-element activation
    residual is available, its weighted L1 sum gives a tighter group radius. -/
theorem w4a8_weighted_residual_error {n : ℕ} (q x xh : Fin n → ℝ) :
    |∑ i, q i * (x i - xh i)| ≤ ∑ i, |q i| * |x i - xh i| := by
  calc
    |∑ i, q i * (x i - xh i)| ≤ ∑ i, |q i * (x i - xh i)| :=
      Finset.abs_sum_le_sum_abs _ _
    _ = ∑ i, |q i| * |x i - xh i| := by
      apply Finset.sum_congr rfl
      intro i _
      rw [abs_mul]

/-- K14a. Certified head screen, completeness. `t` = true logits, `s` = screen scores, `ρ` = radii with
    |t j − s j| ≤ ρ j. `L = s i0 − ρ i0` is the best lower bound. If `tstar` bounds all true logits
    from above, then every token within `w` of the true maximum survives the test `s j + ρ j ≥ L − w`. -/
theorem screen_complete {n : ℕ} (t s ρ : Fin n → ℝ) (tstar L w : ℝ) (i0 : Fin n)
    (hb : ∀ j, |t j - s j| ≤ ρ j) (hstar : ∀ j, t j ≤ tstar) (hL : L = s i0 - ρ i0)
    (j : Fin n) (hj : tstar - w ≤ t j) : L - w ≤ s j + ρ j := by
  have h1 := (abs_le.mp (hb i0)).1
  have h2 := (abs_le.mp (hb j)).2
  have h3 := hstar i0
  linarith

/-- K14b. Tie window. `f` (soft-cap ∘ bf16 rounding) is monotone; `z0` is a lower bound for every `z`
    with `f z ≥ f zL`. If the maximum `zs ≥ zL` and token value `zj` ties with it (`f zj = f zs`), then
    `zj ≥ z0`. The engine computes `z0` by bisection and subtracts a bf16 half-step (0.25) for the
    rounding of the real logit. -/
theorem tie_window (f : ℝ → ℝ) (hmono : Monotone f) (zL zs zj z0 : ℝ) (hLs : zL ≤ zs)
    (hz0 : ∀ z, f zL ≤ f z → z0 ≤ z) (htie : f zj = f zs) : z0 ≤ zj :=
  hz0 zj (by rw [htie]; exact hmono hLs)

/-- K14c. The fp32 accumulation margin 1e-4·‖w‖·‖h‖ covers the dot-product rounding error:
    γ₁₅₃₆ = 1536u/(1−1536u) with u = 2⁻²⁴ is below 1e-4 (and |Σ wᵢhᵢ| ≤ ‖w‖‖h‖ by Cauchy–Schwarz,
    l2_radius_sq). The rounding model itself (standard fl(a∘b)=(a∘b)(1+d)) is an assumption. -/
theorem gamma_1536_le : (1536 : ℝ) * 2 ^ (-24 : ℤ) / (1 - 1536 * 2 ^ (-24 : ℤ)) ≤ 1 / 10000 := by
  norm_num

/-- K15a. Optimal draft-length rule. A round yields E expected tokens in time T (rate E/T). Adding one
    more draft token adds q expected tokens (q = P(all drafts so far and this one accepted)) and costs
    c more time (a draft step plus one more verify row). The rate rises iff q > (E/T)·c. -/
theorem extend_draft_iff (E T q c : ℝ) (hT : 0 < T) (hc : 0 < c) :
    E / T < (E + q) / (T + c) ↔ E / T * c < q := by
  have hTc : 0 < T + c := by linarith
  rw [div_lt_div_iff₀ hT hTc]
  constructor
  · intro h; rw [div_mul_eq_mul_div, div_lt_iff₀ hT]; nlinarith
  · intro h; rw [div_mul_eq_mul_div, div_lt_iff₀ hT] at h; nlinarith

/-- K15b. Expected tokens per round with k drafts: E_k = 1 + ∑ qᵢ, nondecreasing in k when qᵢ ≥ 0, and
    each qᵢ is a product of per-position acceptance probabilities, so qᵢ is nonincreasing in i when
    every factor is ≤ 1. Hence the set of worthwhile draft lengths is a prefix: once a draft fails the
    test q > R·c it never pays to add deeper ones that cost at least as much. -/
theorem q_antitone (a : ℕ → ℝ) (h0 : ∀ i, 0 ≤ a i) (h1 : ∀ i, a i ≤ 1) (i : ℕ) :
    ∏ j ∈ Finset.range (i + 1), a j ≤ ∏ j ∈ Finset.range i, a j := by
  rw [Finset.prod_range_succ]
  exact mul_le_of_le_one_right (Finset.prod_nonneg fun j _ => h0 j) (h1 i)

/-- K16. Split-KV softmax merge. A thread that owns a slice A of the keys keeps its own maximum `mA`
    and the sums `∑ exp(sᵢ − mA)·gᵢ` (g = 1 for the denominator, g = vᵢ for the numerator). Rescaling
    them to any common base `c` is exact: ∑ exp(sᵢ − c)·gᵢ = exp(mA − c) · ∑ exp(sᵢ − mA)·gᵢ.
    Hence the merged softmax-weighted value is identical to the single-pass result. -/
theorem softmax_partial_rescale {n : ℕ} (s g : Fin n → ℝ) (A : Finset (Fin n)) (mA c : ℝ) :
    ∑ i ∈ A, Real.exp (s i - c) * g i = Real.exp (mA - c) * ∑ i ∈ A, Real.exp (s i - mA) * g i := by
  rw [Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro i _
  have : Real.exp (s i - c) = Real.exp (mA - c) * Real.exp (s i - mA) := by
    rw [← Real.exp_add]; congr 1; ring
  rw [this]; ring

/-- K16b. Disjoint slices add: the full sum is the sum over the slices. -/
theorem softmax_split_sum {n : ℕ} (f : Fin n → ℝ) (A B : Finset (Fin n)) (h : Disjoint A B) :
    ∑ i ∈ A ∪ B, f i = ∑ i ∈ A, f i + ∑ i ∈ B, f i := Finset.sum_union h

end CpuLaptop

#print axioms CpuLaptop.screen_keeps_argmax
#print axioms CpuLaptop.head_saving_iff
#print axioms CpuLaptop.kv_at_32k
#print axioms CpuLaptop.kv_per_token_long
#print axioms CpuLaptop.lookahead_comp
#print axioms CpuLaptop.prefix_stable
#print axioms CpuLaptop.spec_needs_alpha_gt_c
#print axioms CpuLaptop.ple_resident
#print axioms CpuLaptop.slook_comp
#print axioms CpuLaptop.audio_subsampler_lookahead
#print axioms CpuLaptop.spec_k1_iff
#print axioms CpuLaptop.ollama_bytes
#print axioms CpuLaptop.stock_bytes
#print axioms CpuLaptop.spec_beats_ollama_bytes
#print axioms CpuLaptop.anon_bytes_after_ple_map
#print axioms CpuLaptop.l2_radius_sq
#print axioms CpuLaptop.act_quant_radius
#print axioms CpuLaptop.body_time
#print axioms CpuLaptop.spec_round_flat
#print axioms CpuLaptop.bf16_fields_unique
#print axioms CpuLaptop.codebook_roundtrip
#print axioms CpuLaptop.packed_head_bytes
#print axioms CpuLaptop.budget_head_cap
#print axioms CpuLaptop.packed_head_throughput_lt_31
#print axioms CpuLaptop.patch_exact
#print axioms CpuLaptop.w4a8_no_overflow
#print axioms CpuLaptop.w4a8_offset_identity
#print axioms CpuLaptop.w4a8_group_error
#print axioms CpuLaptop.screen_complete
#print axioms CpuLaptop.tie_window
#print axioms CpuLaptop.gamma_1536_le

#print axioms CpuLaptop.extend_draft_iff
#print axioms CpuLaptop.q_antitone

#print axioms CpuLaptop.softmax_partial_rescale
#print axioms CpuLaptop.softmax_split_sum
