# Method: how I worked

This repository is the product of a loop, and the loop matters more than any single trick. I steered an AI coding assistant with a small set of rules. This page lists the rules, what each one found, and the mistakes it caught.

## The loop

1. **Find the physical limit first.** Before writing any code, measure what the machine can do and compute what the model needs.
2. **Pose each problem as a math problem.** Write down what is being optimised and what the constraints are, and look for a known result before inventing something.
3. **Write the C++ in a low-latency style.** No allocation in the token loop, cache-line-aligned per-thread state, a persistent thread pool that spins briefly and then sleeps, non-temporal prefetch for streamed weights.
4. **Prove the core claim in Lean** when it is arithmetic or logic. Not everything can be proved (speed cannot), but "this error bound keeps the argmax exact" can.
5. **Measure against a fair baseline in the same session.** Alternate the two builds, rotate the order, and check that the generated tokens are identical when the change is supposed to be exact.
6. **Write down every failure.** The experiments that did not work are in [RESULTS.md](RESULTS.md), with numbers, so nobody repeats them.

## Step 1: the physical limit

Decoding one token reads every weight once. On this laptop a plain AVX2 read loop streams about **21.3 GB/s** whatever the page size (4 KB pages, 2 MB huge pages, a mapped file all give 20.5 to 21.6 GB/s). Six threads reach 20 to 21 GB/s, eight threads about 19.3. The int4 body of the model, with its scales, is about 1,056 MB, and the screened output head adds 226 MB. So:

- body only: 1,056 MB / 21.3 GB/s = **49.6 ms** per forward pass
- body and head: about **60 to 64 ms** per forward pass

Decode speed is then `tokens per round / forward time`. Two levers follow immediately. Reduce the bytes read per pass (the head screen, int4 everywhere), and get more tokens out of each pass (speculative decoding). Nothing else can matter much once the kernels sit at the bandwidth ceiling, and that is where they ended up: the matrix products measure 51.2 ms against a 48.5 ms floor.

`bench/stream_bw.cpp` measures the ceiling on your own machine. Run it first.

## Step 2: math problems and what they said

| question | result | Lean |
|---|---|---|
| Is a drafted token worth its cost? | Speculation only pays when the acceptance rate exceeds the cost ratio of drafting. | `spec_needs_alpha_gt_c`, `spec_k1_iff` |
| Should the drafter also propose a second candidate (a sibling) instead of a longer chain? | The sibling beats the chain exactly when `q2 > p1 * p2`. Measured: the drafter's top guess is right 59.8% of rounds and its second guess only 16.7%, so an unconditional sibling does not win. | `sibling_beats_chain_iff`, `sibling_rate_iff` |
| Can the 262,144-row output matrix be read less often? | Yes. Screen with a small int4 copy and a radius from Cauchy-Schwarz; the true argmax can never be discarded, so the output stays exact. The screen is about 98% of the head's time. | `screen_keeps_argmax`, `act_quant_radius`, `head_saving_iff` |
| Could a cheaper (3-bit or cluster) screen work? | No. 99.6% of tokens survive the cluster bound, because the head rows are nearly orthogonal in 1,536 dimensions. Measured, not proved. | none |
| Does the int8 activation path overflow its 16-bit accumulator? | No: 4 chunks of at most 7,620 add to at most 30,480, below 32,767. | `w4a8_no_overflow`, `w4a8_offset_identity`, `w4a8_group_error` |
| Can a packed bf16 head reach 31 tok/s? | No, by arithmetic: two tokens per round at 21.5 GB/s over 1,131 MB of body and drafter plus a 554 MB packed head is under 31. That is why the screen exists. | `packed_head_throughput_lt_31`, `budget_head_cap` |
| Can partial attention results from different threads be merged exactly? | Yes, with a log-sum-exp rescale of each thread's partial softmax. Checked by comparing the engine's tokens with the unsplit path. | none in this repository |
| Can the screen's per-thread winners be combined in any order? | Yes, the best candidate of partial results is order-independent. | `partial_winner_associative` |
| How much lag can the HTTP layer add? | At most the sum of its stage times, and no queue forms while bursts are at least that far apart. | `stage_lag_le`, `two_stage_lag_le` |
| Can streaming a stop string retract text already sent? | Not if you hold back one character fewer than the stop string. | `holdback_safe`, `slices_concat` |
| Is reusing cached keys and values after a slot switch correct? | Yes, for a causal model, below the longest common prefix. The chosen slot maximises the shared prefix, and evicting never loses the active conversation. | `reuse_sound`, `best_slot_max`, `evict_parks_active` |
| How long should an idle worker spin before it sleeps? | Spin for a budget longer than any gap inside a request; then the worst case is within `1 + tau/W` of optimal and nothing sleeps mid-request. | `park_competitive`, `no_park_in_request` |

Everything in the table is in [`proofs/`](../proofs). The proofs check the arithmetic and logic. They do not check that the C++ implements the statement, which I verified by running the engine against reference outputs instead.

## Step 3: low-latency C++

The practices that mattered, in the order they paid off:

- the weight kernels stream with `prefetchnta`, so weights that will not be reused do not evict the cache lines that will
- one persistent thread pool; workers spin with `_mm_pause` and sleep after a budget, and a dispatch publishes a job word that includes the sequence number and the active thread count
- the thread count adapts: 6 threads for short contexts, 8 once the context passes 1,024 tokens
- per-thread state is aligned to 64 bytes so threads never share a cache line
- no allocation, no locks and no system calls in the token loop; memory is allocated up front and optionally locked
- work is split 7/8 static and 1/8 dynamic, so one slow thread cannot stall the end of a matrix product

## Step 5: measuring fairly

This is the rule that saved the most time, because this laptop's speed drifts by about **10%** during a session (clocks, thermals, other programs).

- Compare two builds only in the same session, alternating A and B and rotating the order. `bench/ab.py` does exactly this.
- A change that is supposed to be exact must produce **identical tokens**. The A/B script checks.
- Anything under about 3 to 5% is noise here. I stopped chasing such changes.
- Set an absolute gate as well as a relative one: a relative gain on a prompt that runs at 5 tokens per second is not progress.
- Do not compile or run anything else during a benchmark. A compile in the background cost me one whole run.
- Look at `top`, not at a process list from inside a sandbox, for leftover engines.

## Mistakes the loop caught

I am listing these because they are how the method earns its keep.

- **An idle engine spinning at 500% CPU.** The thread pool never slept. It showed up only when two engines were alive at once and the second one ran slower. Fixed with the spin-then-sleep budget.
- **A hang on any prompt longer than 1,024 tokens.** A worker waking from sleep overwrote its "last seen job" counter and skipped a job that was already published. The adaptive-threads feature had been blamed for it and nearly removed.
- **Single-prompt wins quoted as general.** One prompt gave 1.44 times Ollama; across six it was 1.08. Every claim now needs the whole table.
- **Stale baselines.** Comparing today's run with last week's number is not a comparison.
- **My own arithmetic.** I (and the assistant) once reported a mean of 14.67 where the sum gives 14.80.
- **Bad settings for the competitor.** My first stock-vLLM run had all threads pinned to one CPU and the next run was killed for running out of memory. A baseline that is badly configured is not a result.

## Learning from vLLM and Ollama

I read how both serve. The ones I took: llama.cpp's spin-then-sleep polling (`--poll`), the longest-common-prefix slot choice that llama.cpp and vLLM's prefix cache both use, vLLM's incremental detokeniser that decodes only a short window per token, and the `cached_tokens` field in the usage block. The one I rejected is batching several tokens per streamed chunk, because the engine already emits one chunk per decode round.
