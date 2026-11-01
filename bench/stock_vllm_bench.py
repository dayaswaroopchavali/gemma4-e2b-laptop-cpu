"""Stock vLLM (CPU build) with the best settings I found, on the same prompts: compiled (not eager), the MTP drafter at k=3, one sequence, 8 threads pinned to physical cores, 1 GiB KV cache.
Run it with the Python of a vLLM CPU install:   VLLM_CPU_OMP_THREADS_BIND=0,2,4,6,8,10,12,14 VLLM_CPU_KVCACHE_SPACE=1 python stock_vllm_bench.py
(Do NOT use VLLM_CPU_OMP_THREADS_BIND=auto on a 2-way SMT laptop: it resolved to a single CPU here. A 4 GiB KV cache got the worker OOM-killed on 15 GB of RAM.)"""
import statistics, time
from vllm import LLM, SamplingParams
from common import *
llm = LLM(str(CKPT), dtype="bfloat16", max_model_len=8192, enforce_eager=False, max_num_seqs=1, enable_prefix_caching=False, limit_mm_per_prompt={"image": 0, "audio": 0, "video": 0},
          disable_log_stats=True, speculative_config={"model": str(DRAFTER), "num_speculative_tokens": 3}); eng = llm.llm_engine; rid = 0
def one(ids, n):
    global rid; rid += 1; eng.add_request(f"r{rid}", {"prompt_token_ids": ids}, SamplingParams(temperature=0, max_tokens=n)); t0 = time.time(); tf = nf = None
    while eng.has_unfinished_requests():
        for o in eng.step():
            if o.request_id != f"r{rid}": continue
            toks = o.outputs[0].token_ids
            if tf is None and toks: tf = time.time(); nf = len(toks)
            if o.finished: return len(toks), nf, tf - t0, time.time() - tf
one([int(x) for x in prompt_ids(prompt_names()[0]).split()[:200]], 16)      # warm-up (compilation)
print(f"{'prompt':16s} {'tokens':>6s} {'first output s':>15s} {'decode tok/s':>13s}")
for n in prompt_names():
    ids = [int(x) for x in prompt_ids(n).split()]; k = 60 if n in ("document_6800", "repeat_1800") else 120; r = []
    for rep in range(2): total, nf, ttft, tdec = one(ids, k); r.append((total - nf) / tdec if tdec > 0 and total > nf else float("nan"))
    print(f"{n:16s} {len(ids):6d} {ttft:15.2f} {statistics.mean(r):13.2f}", flush=True)
