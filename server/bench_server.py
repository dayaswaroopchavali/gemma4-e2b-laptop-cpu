"""Decode tok/s and time to first token through the HTTP server (streaming).  usage: bench_server.py [base_url]"""
import sys, time, statistics, pathlib
from openai import OpenAI
ROOT = pathlib.Path(__file__).resolve().parent.parent; P = ROOT / "bench" / "prompts"; base = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
c = OpenAI(base_url=base + "/v1", api_key="EMPTY"); M = c.models.list().data[0].id
print(f"{'prompt':16s} {'cold first token s':>19s} {'warm first token ms':>20s} {'decode tok/s':>13s}")
for f in sorted(P.glob("*.txt")):
    if f.stem == "repeat_1800": continue
    txt = f.read_text(); n = 60 if f.stem == "document_6800" else 120; r = []
    for rep in range(2):
        t0 = time.time(); tf = tl = u = None
        for ch in c.chat.completions.create(model=M, messages=[{"role": "user", "content": txt}], max_tokens=n, stream=True, stream_options={"include_usage": True}):
            if ch.choices and ch.choices[0].delta.content: tl = time.time(); tf = tf or tl
            if not ch.choices: u = ch.usage
        r.append((tf - t0, (u.completion_tokens - 1) / (tl - tf) if tl > tf else float("nan")))
    print(f"{f.stem:16s} {r[0][0]:19.2f} {r[1][0]*1000:20.0f} {statistics.mean(x[1] for x in r):13.2f}", flush=True)
