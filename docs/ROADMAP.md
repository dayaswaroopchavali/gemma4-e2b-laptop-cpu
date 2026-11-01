# What is left, and what I would do next

## The one number I did not reach

My own minimum was 25 tokens per second on every prompt. The slowest prompt (6,841 tokens) is at 24.4, which is within the noise of this laptop but not over the line. Every part of the forward pass is already near its memory floor (see [RESULTS.md](RESULTS.md)), so another kernel trick cannot close it. The only lever left is **more accepted tokens per round**, which means improving the drafter.

1. **Fine-tune the MTP drafter on the target text.** The drafter's top guess is right 59.8% of the time. Raising that raises tokens per round directly. It needs a GPU for a short training run, and it is empirical rather than provable, so it sits outside my "prove it in Lean" rule. I have not done it.
2. **An int8 or int4 drafter.** The drafter costs 3 to 4 ms per forward pass. Quantising its weights could save about 1 ms. Small, but free.
3. **A lower bytes-per-token model or a smaller head.** The screened head is 226 MB of the roughly 1,280 MB read per pass.

## Server work

The server covers text chat and completions. Not done, in the order I would do them:

1. **Audio input.** The engine already has an audio tower and an `AUDIO_FEED` command in its line protocol. Wire it to the `input_audio` content part, and stream audio into the engine while the speaker is still talking so only the last chunk is processed after they stop.
2. **Tool-call parsing.** Gemma-4 has its own tool-call format. The chat template already renders tools, but the output is returned as plain text.
3. **Several callers at once.** The engine serves one request at a time, and audio state is global. Real concurrency needs per-caller audio state and a scheduling policy for one set of cores.
4. **Sampling.** Decoding is greedy because the head screen certifies the argmax. Sampling would need real logits for a candidate set; a top-k variant of the screen is the obvious route.
5. **`n > 1`, `logprobs`.** Same reason as sampling.

## Portability

- **Zen 4 and newer Intel CPUs have VNNI and AVX-512.** The int8 path could use `vpdpbusd` and would likely be much faster. The kernels are written for AVX2 because that is all this laptop has.
- **ARM (NEON, SVE).** Not attempted. The memory-bandwidth argument is the same, so the design should carry over.
- **Other models.** The engine is written for Gemma-4 E2B only. The ideas (certified head screen, speculative decoding with a small drafter, int8 activations over int4 weights) are not specific to it.

## Measurement hygiene I would keep

- One laptop is one data point. A second machine would say how much of this is the engine and how much is the machine.
- Memory locking and the keep-warm thread are argued from a measured refault cost and have not been tested under real memory pressure.
