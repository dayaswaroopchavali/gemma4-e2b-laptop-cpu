# Step by step

Everything here was done on Linux with an AMD Ryzen 7 7730U laptop. Where your machine differs, the notes say what to change.

## 0. What you need

- an x86-64 CPU with **AVX2 and FMA** (the kernels also use `-march=native`); about 12 GB of free RAM; about 12 GB of disk for the weights
- `g++` 11 or newer, Python 3.10 or newer
- optional, for the comparisons: [Ollama](https://ollama.com) with `gemma4:e2b`, and a CPU build of [vLLM](https://github.com/vllm-project/vllm)
- optional, for the proofs: Lean 4 with Mathlib (`elan`, then `lake`)

```bash
python -m venv .venv && . .venv/bin/activate
pip install transformers fastapi uvicorn openai huggingface_hub numpy matplotlib
```

## 1. Get the models

Two folders under `models/`. Both are Google's, under the Gemma terms, and neither is in this repository.

```bash
huggingface-cli download google/gemma-4-E2B-it-qat-w4a16-ct --local-dir models/gemma-4-E2B-it-qat
huggingface-cli download google/gemma-4-E2B-it-assistant     --local-dir models/gemma-4-E2B-it-assistant
```

The first is the int4 quantisation-aware-trained checkpoint (compressed-tensors, group size 32). The second is the small MTP drafter used for speculative decoding. If the first name has changed, look in Google's Gemma-4 QAT collection on Hugging Face for the `w4a16` checkpoint of E2B.

## 2. Build

```bash
./build.sh                      # -> bin/gemma4_engine
python tools/make_index.py models/gemma-4-E2B-it-qat/model.safetensors       models/engine_index.tsv
python tools/make_index.py models/gemma-4-E2B-it-assistant/model.safetensors models/drafter_index.tsv
```

The index files list every tensor with its offset in the safetensors file, so the engine can map the weights directly.

## 3. Measure your machine first

```bash
g++ -O3 -march=native bench/stream_bw.cpp -o /tmp/stream_bw -lpthread && for t in 1 2 4 6 8; do /tmp/stream_bw $t; done
```

This prints the read bandwidth for each thread count (threads are pinned to every second logical CPU; set `BW_STRIDE=1` if your cores are numbered differently). Divide 1,056 MB by the best number to get the floor for one forward pass. If your machine streams faster, expect a faster engine.

## 4. Pick the cores

`bench/common.py` pins the engine to `0,2,4,...,14` through `ENGINE_CPUS`. That is one logical CPU per physical core on a laptop whose SMT siblings are numbered 0/1, 2/3, and so on. Check yours with `lscpu -e` and set `ENGINE_CPUS` to one logical CPU per physical core. Using both siblings of a core slows decode here.

## 5. Make prompts and run

```bash
python bench/make_prompts.py        # writes bench/prompts/*.txt and *.ids (needs the tokenizer in models/)
python bench/run_engine.py          # decode tok/s and cold prefill per prompt
python bench/int8_vs_exact.py       # what the default int8 decode costs in tokens
python bench/ab.py base=bin/gemma4_engine exact=bin/gemma4_engine:ENGINE_DECODE_INT8=0
```

The engine reads requests as lines on its standard input (`ENGINE_SERVE=1`): `N <prompt token ids>`, where `N` is the number of new tokens. It prints `FIRST <id>` after the prefill, then `GEN: ...` and a `TIMING` line. With `ENGINE_STREAM=1` it also prints `TOK <ids>` after each decode round, which is what the server uses to stream.

Useful settings, all environment variables:

| variable | default | meaning |
|---|---|---|
| `ENGINE_K` | 3 | number of drafted tokens per round |
| `ENGINE_DECODE_INT8` | on | `0` switches to exact float decode |
| `ENGINE_ADAPTIVE_THREADS`, `ENGINE_THREAD_CUTOFF` | off, 1024 | use 6 threads, and 8 once the context passes the cutoff |
| `ENGINE_CPUS` | 0,2,4,6,8,10 | cores to pin the pool to |
| `ENGINE_SLOTS` | 1 | prefix-cache slots (about 150 MB each at 8k context) |
| `ENGINE_PARK_US` | 2000 | how long an idle worker spins before it sleeps |
| `ENGINE_MLOCK`, `ENGINE_KEEPWARM` | off | lock anonymous memory; re-fault evicted weight pages while idle |

## 6. Compare with Ollama and stock vLLM

```bash
python bench/ollama_bench.py gemma4:e2b
VLLM_CPU_OMP_THREADS_BIND=0,2,4,6,8,10,12,14 VLLM_CPU_KVCACHE_SPACE=1 python bench/stock_vllm_bench.py   # in a venv with vLLM CPU
```

Run all three in the same session, back to back, and do not run anything else meanwhile. Two settings cost me a failed run each on stock vLLM: `VLLM_CPU_OMP_THREADS_BIND=auto` pinned every thread to a single CPU, and a 4 GiB KV cache made the process run out of memory on 15 GB.

## 7. Serve it

```bash
python server/server.py serve --port 8000 --served-model-name gemma-4-e2b
python server/test_server.py http://localhost:8000     # 25 checks with the official OpenAI client
python server/bench_server.py http://localhost:8000
curl localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"gemma-4-e2b","messages":[{"role":"user","content":"Say hi in five words."}],"stream":true}'
```

`vllm serve`-style flags that the engine does not use are accepted and ignored with a warning. Flags that matter: `--slots 4` (conversations kept), `--max-model-len 8192`, `--engine-cpus`, `--no-mlock`, `--exact`.

## 8. Charts and proofs

```bash
python bench/make_charts.py                       # rewrites charts/*.png from the numbers inside it
cd proofs && lake env lean ServerLag.lean && lake env lean SlotCache.lean   # inside a Lean project that depends on Mathlib
```

The proofs import Mathlib, so they have to be compiled inside a Lake project that requires it. Lean version 4.35 with the matching Mathlib release is what they were written against.

## If something looks wrong

- Decode speed halves while another heavy process runs: pin it elsewhere, or stop it. The engine's threads spin and are sensitive to being preempted.
- The first request after a long idle is slower: the weights were evicted from memory. Raise `ulimit -l` and use `ENGINE_MLOCK=1`.
- Your numbers differ from mine by 10%: that is normal. Run `bench/ab.py` instead of comparing across sessions.
