# Native Gemma-4-E2B decode engine — exact spec extracted from vLLM 0.32 (oracle)

Goal: single-stream, batch-1 C++/AVX2 engine for ONE model (google/gemma-4-E2B-it-qat-w4a16-ct) on ONE CPU
(Ryzen 7 7730U), reusing the verified int4 g32 kernel (csrc/cpu/w4g32_avx2.h), the packed head
(csrc/cpu/bf16_exp3_head.h) and Google's MTP drafter. vLLM stays the reference: every stage is checked
against recorded vLLM activations before the next stage is written.

## Why (measured)
Plain vLLM step ~125 ms/token = ~21 ms non-matmul (Python/eager/scheduler/attention/norms) + ~100 ms matmuls
(body+head at ~16 GB/s of a ~21.5 GB/s ceiling). Byte floor ~75 ms/token plain; Ollama's 22 tok/s is speculation
(n-max 3, acceptance 0.35) at ~zero overhead. Target needs overhead ~0, head < ~256 MB, ~2 tok/round (Lean: budget_head_cap).

## Dimensions (config)
35 layers, hidden 1536, MLP 6144 (all layers), vocab 262144 (tied head/embed), PLE dim 256/layer, 8 q heads, 1 kv head,
layer_types: full_attention at layers 4,9,14,19,24,29,34 else sliding (window 512); head_dim 256 sliding / 512 full;
last 20 layers (15..34) KV-shared: layer L reuses K/V of the last owner of the same type among layers 0..14
(sliding -> 13, full -> 14); shared layers have only q_proj (no k/v proj, no k_norm/v_norm).
RoPE: sliding theta 1e4 rope_type default full rotary; full theta 1e6 rope_type proportional, partial_rotary_factor 0.25;
NeoX style. attention scaling = 1.0 (q_norm/k_norm carry scale), no attn softcap. final_logit_softcapping 30.0.
rms_norm_eps 1e-6.

## Per-token input
h0 = embed_tokens[tok] * normalizer (= sqrt(1536))                          [bf16 table, tied with head]
ple_embed = embed_tokens_per_layer[tok] * embed_scale_per_layer (= sqrt(256)) reshaped [35,256]   [4.7 GB table, row gather]
proj = per_layer_model_projection(h0) * per_layer_projection_scale (= 1536^-0.5), reshape [35,256], RMSNorm(per_layer_projection_norm)
per_layer_inputs = (proj + ple_embed) * per_layer_input_scale (= 2^-0.5)

## Decoder layer l (hidden_states x, per_layer_input p_l)
r = x
a = input_layernorm(r)
 owner layers:  qkv = qkv_proj(a) -> q [8*hd], k [hd], v [hd]
                q = q_norm(q per head); k = k_norm(k per head); v = v_norm(v per head, NO weight)
                (q,k) = rope(positions, q, k); write k,v to cache
 shared layers: q = q_proj(a); q = q_norm(q per head); q = rope(q); K/V from owner layer cache
 attn: causal; sliding layers limited to last 512 positions; scaling 1.0; GQA 8:1 -> o_proj
x = post_attention_layernorm(attn_out) + r ; r = x
m = mlp(pre_feedforward_layernorm(x)) : gate_up_proj -> gelu_tanh(gate) * up -> down_proj
x = post_feedforward_layernorm(m) + r
PLE: g = gelu_tanh(per_layer_input_gate(x)) * p_l ; x = x + post_per_layer_input_norm(per_layer_projection(g))
x = x * layer_scalar
(RMSNorm weight form to be verified from checkpoint values before coding: vLLM comment says output = norm(x) * weight)

## Output
x = final norm(x); logits = x @ embed_tokens^T ; logits = 30 * tanh(logits / 30); greedy argmax (lowest index on ties).

## Corrections found by validating against vLLM (pyref_gemma4.py reproduces all 35 layers + first token)
- MLP width is NOT 6144 everywhere: layers 0-14 use 6144; layers 15-34 (the KV-shared ones) use 12288 (double-wide).
- RMSNorm is x * rsqrt(mean(x^2)+1e-6) * weight (no 1+weight). v_norm has no weight. Residual stream is bf16 between layers.
- Proportional RoPE (full-attention layers, head_dim 512, theta 1e6, partial 0.25): inv_freq[j] = theta^(-2j/head_dim) for the first
  rope_angles = 64 pair-indices j, ZERO (identity) for the remaining 192 pair-indices; NeoX pairing i <-> i+head_dim/2 over the WHOLE head.
  Sliding layers (theta 1e4, head_dim 256): same formula with all 128 pair-indices rotating (standard NeoX).
- per_layer_model_projection is int4 g32 ([8960,1536]); per-layer gates/projections are int4 too; embed_tokens_per_layer is a [262144, 8960] bf16 row table.
- Verified: per-layer max relative error vs vLLM <= 1.6% (bf16 level, not growing); final-norm 3.3%; first generated token 236770 matches (top logit 28.74 vs runner-up 25.31).

## Build order (each stage verified bit-/tolerance-wise against recorded vLLM tensors)
1. dump reference tensors from vLLM eager for a fixed prompt (embeddings, PLE inputs, per-layer outputs, logits).
2. C++ RMSNorm/rope/gelu primitives vs reference.  3. one owner layer. 4. one shared layer. 5. all 35 layers + head.
6. KV cache + attention for decode (sliding window) and prefill.  7. speculative verify loop with the MTP drafter.
8. certified head screen. 9. audio embedding input (external encoder) and streaming prefill.
