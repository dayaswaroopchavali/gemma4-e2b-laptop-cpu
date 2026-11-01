"""Quantify what int8 decode costs: for each prompt, how far the int8 tokens follow the exact-float tokens before the first difference."""
from common import *
print(f"{'prompt':16s} {'tokens':>7s} {'first difference at':>20s}")
for n in prompt_names():
    k = 60 if n in ("document_6800", "repeat_1800") else 120
    a = run_engine(n, k, env=engine_env())["tokens"]; b = run_engine(n, k, env=engine_env(ENGINE_DECODE_INT8=0))["tokens"]
    d = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
    print(f"{n:16s} {min(len(a), len(b)):7d} {'identical' if d is None else f'token {d}':>20s}", flush=True)
