# Results

All numbers are from one laptop (Ryzen 7 7730U, 8 cores, about 15 GB RAM, no GPU), greedy decoding, one stream. The model is the same Gemma-4 E2B int4 for all three systems, and all three use Google's MTP drafter. Each decode number is the mean of two runs.

## Decode speed (tokens per second, higher is better)

| prompt | tokens | Ollama | stock vLLM | this engine | engine / Ollama |
|---|---:|---:|---:|---:|---:|
| repeated text | 1,796 | 24.01 | 21.07 | 59.98 | 2.50 |
| long document | 6,841 | 17.54 | 9.94 | 24.41 | 1.39 |
| message summary | 165 | 22.42 | 12.20 | 29.94 | 1.34 |
| alert text | 57 | 22.42 | 16.83 | 37.45 | 1.67 |
| call request | 37 | 29.24 | 14.63 | 43.77 | 1.50 |
| short command | 18 | 19.79 | 12.61 | 30.96 | 1.56 |
| tool-call JSON | 71 | 29.23 | 19.90 | 38.21 | 1.31 |
| running summary | 581 | 24.18 | 11.19 | 28.46 | 1.18 |
| **mean** | | **23.60** | **14.80** | **36.65** | **1.55** |

Without the repeated-text prompt, which flatters speculative decoding, the engine's mean advantage over Ollama is 1.42 times. The engine is slowest relative to Ollama on the 581-token prompt (1.18 times).

## Prefill (seconds from a cold prompt to the first token, lower is better)

| prompt | tokens | Ollama | stock vLLM | this engine |
|---|---:|---:|---:|---:|
| repeated text | 1,796 | 17.1 | 31.3 | 12.3 |
| long document | 6,841 | 73.9 | 139.0 | 58.4 |
| message summary | 165 | 1.43 | 3.00 | 1.15 |
| alert text | 57 | 0.51 | 1.25 | 0.41 |
| call request | 37 | 0.36 | 0.90 | 0.28 |
| short command | 18 | 0.23 | 0.50 | 0.14 |
| tool-call JSON | 71 | 0.65 | 1.50 | 0.51 |
| running summary | 581 | 5.27 | 10.35 | 4.50 |

Stock vLLM's figure is time to the first output, so it includes one decode step. The engine's prefill uses a bounded int8 path that can change a token late in a few outputs.

## Caveats you should know about

- **The prompts were private.** I measured on prompts of these lengths, but with my own text. The prompts in `bench/prompts/` are neutral stand-ins of similar lengths (20 to 6,855 tokens). Speculative decoding accepts more tokens on repetitive text, so your numbers will differ, in either direction.
- **Ollama was measured right after the engine runs, not interleaved.** The laptop drifts about 10% in a session, so treat differences under 5% as equal. Ollama's own runs varied (22.4 to 25.6 on the repeated text, 16.3 to 18.8 on the long document).
- **Ollama generated 120 tokens on the short command where the engine stopped at 58**, so that pair is less comparable.
- **Stock vLLM was tuned as well as I could:** compiled rather than eager, the same drafter at k=3, eight pinned threads, a 1 GiB KV cache. I could not use tcmalloc (not installed).

## What the int8 decode costs

Int8 decode is the default. It is exact in its arithmetic (an overflow bound is proved), but it quantises the activations, so it is not bit-identical to the float kernel. I compared both against the reference tokens from stock vLLM, each run to its stop token:

| prompt | tokens | int8 matches reference | float matches reference | int8 tok/s | float tok/s |
|---|---:|---:|---:|---:|---:|
| message summary | 38 | 38 | 38 | 29.3 | 25.9 |
| alert text | 43 | 43 | 43 | 36.9 | 31.4 |
| call request | 18 | 18 | 18 | 44.0 | 35.7 |
| short command | 58 | 53 | 58 | 31.0 | 27.0 |
| tool-call JSON | 34 | 34 | 34 | 37.1 | 31.1 |
| running summary | 53 | 53 | 24 | 28.6 | 22.5 |

Five of six prompts follow the reference to the end under int8, and five of six under float. Each mode diverges once, at a near-tie token, and on a different prompt. Int8 is 13 to 27% faster. The sample is six prompts and 244 tokens, so this is a measurement, not a tight bound. `bench/int8_vs_exact.py` reruns it on your prompts.

## Where one forward pass goes

The running-summary prompt (581 tokens), 72.9 ms per forward pass:

| part | ms | share |
|---|---:|---:|
| int4 matrix products | 51.2 | 70% |
| output head | 13.2 | 18% |
| drafter | 3.3 | 5% |
| attention | 2.7 | 4% |
| everything else | 2.5 | 3% |

The long document (6,841 tokens), 81.5 ms: matrices about 51, head 11.6, attention 9.8, drafter 4.4, other about 5. Every part sits within about 25% of its memory floor. The matrix products are at 95% of it (51.2 against 48.5 ms). A full-attention layer over 6,800 tokens takes 0.88 ms against 0.84 ms for just reading the same bytes.

## Decisions the data made

- **Adaptive threads** (6, then 8 past 1,024 tokens): +4.5% mean across nine cases, up to +15% on long contexts, no change on short ones.
- **Int8 decode and three drafted tokens**: another +11% over the adaptive float engine.
- **A stack of small changes** (dynamic work split in matrices and the head, the drafter on the same thread count as the forward pass): +4.0% mean, individually below noise.
- **Prefix cache with 4 slots**: a returning conversation reuses 580 of 581 tokens, and its first token arrives in about 70 ms instead of 4.5 s. Over an 8-request alternating test the total prefill dropped from 16.1 s to 5.6 s.
- **Idle workers sleep after 2 ms**: an idle engine used about 500% CPU before, and 0% after, with no change in decode speed.

## What did not work

Each of these was measured and dropped. They are the most reusable part of the project.

| idea | result |
|---|---|
| feed the drafter pre-norm activations | worse |
| tiled prefill attention | at most 5% faster below 1,800 tokens, not bit-exact |
| sibling candidates and drafter-knob sweeps on the 581-token prompt | nothing beat the default |
| a cluster-bound or 3-bit head screen | 99.6% of tokens survive the bound |
| lossless packing of the bf16 scales | bit-identical, no speedup |
| huge pages | no effect |
| cache blocking the weight stream | 5 to 15% slower |
| bf16 activation storage | exact but 40% slower (the kernel is bound by micro-ops) |
| more drafter probes (`nprobe` 64 or 128) | no acceptance gain |
| signed-nibble decode | helps only 4 rows at a time |
| software prefetch of keys and values in attention | bit-exact but 13% slower at every distance |
| dynamic matrix split on its own | bit-exact, no measurable gain |
| entropy coding of the int4 bytes | at most 4.25% fewer bytes |
| a 6-thread pool for everything | 8 to 13% slower on long prompts than adaptive threads |

## Serving through HTTP

- The decode rate a client sees equals the engine's own rate (ratio 1.000). The HTTP layer adds no measurable lag, which the proofs in `ServerLag.lean` say it cannot while decode rounds are tens of milliseconds apart.
- Turning on slots, memory locking, the keep-warm thread and streaming output each leave throughput within 1.5%.
- Reusing a cached prompt can differ from a fresh prefill at a near-tie token, because the reused path re-runs the last token as a single row. Any prefix cache has this effect. Switching between slots is bit-exact.
