"""Balanced A/B of engine builds / settings on all prompts (this laptop drifts ~10% over a session, so never compare against an old run).
Each config is  name=binary[:ENV=val,ENV=val]; the order rotates forward then backward.  Reports mean tok/s, ratios vs the first config, and whether tokens are identical.
example: ab.py base=bin/gemma4_engine exact=bin/gemma4_engine:ENGINE_DECODE_INT8=0"""
import statistics, sys
from common import *
cfgs = []
for a in sys.argv[1:]:
    n, v = a.split("=", 1); b, _, e = v.partition(":"); cfgs.append((n, b, dict(kv.split("=") for kv in e.split(",") if kv)))
names = prompt_names(); new = lambda n: 60 if n in ("document_6800", "repeat_1800") else 120
res = {(c[0], n): [] for c in cfgs for n in names}; toks = {}
for order in (list(range(len(cfgs))), list(reversed(range(len(cfgs))))):
    for ci in order:
        name, binary, extra = cfgs[ci]
        for n in names:
            r = run_engine(n, new(n), binary=binary, env=engine_env(**extra)); res[(name, n)].append(r["decode_tok_s"]); toks.setdefault((name, n), r["tokens"])
            print(f"{name:10s} {n:16s} {r['decode_tok_s']:6.2f}", flush=True)
print("\nprompt           " + " ".join(f"{c[0]:>10s}" for c in cfgs) + "   ratio vs first / tokens")
for n in names:
    m = [statistics.mean(res[(c[0], n)]) for c in cfgs]
    print(f"{n:16s} " + " ".join(f"{x:10.2f}" for x in m) + "   " + " ".join(f"{x/m[0]:.3f}" for x in m[1:]) + "  " + " ".join("same" if toks[(c[0], n)] == toks[(cfgs[0][0], n)] else "DIFF" for c in cfgs[1:]))
