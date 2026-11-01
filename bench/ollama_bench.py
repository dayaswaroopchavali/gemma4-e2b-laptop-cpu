"""Ollama decode tok/s and cold prefill on the same prompts (model gemma4:e2b by default). Prompt texts come from bench/prompts/*.txt; the long repetitive prompt is sent raw.
usage: ollama_bench.py [model]"""
import json, sys, statistics, urllib.request
from common import *
model = sys.argv[1] if len(sys.argv) > 1 else "gemma4:e2b"
def post(path, body): return json.load(urllib.request.urlopen(urllib.request.Request("http://localhost:11434/api/" + path, json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=3000))
print(f"{'prompt':16s} {'tokens':>6s} {'prefill s':>10s} {'decode tok/s':>13s}")
for n in prompt_names():
    text = (PROMPTS / f"{n}.txt").read_text(); k = 60 if n in ("document_6800", "repeat_1800") else 120; opts = {"num_predict": k, "temperature": 0, "num_ctx": 8192}
    rates = []; pre = None
    for rep in range(2):
        d = post("generate", {"model": model, "prompt": text, "stream": False, "raw": True, "options": opts}) if n == "repeat_1800" else post("chat", {"model": model, "messages": [{"role": "user", "content": text}], "stream": False, "think": False, "options": opts})
        if rep == 0: pre = d["prompt_eval_duration"] / 1e9      # first call is the cold prefill (later calls hit Ollama's prompt cache)
        rates.append(d["eval_count"] / d["eval_duration"] * 1e9)
    print(f"{n:16s} {d['prompt_eval_count']:6d} {pre:10.2f} {statistics.mean(rates):13.2f}", flush=True)
