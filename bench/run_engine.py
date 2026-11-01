"""Decode speed and cold prefill time of the engine on every prompt in bench/prompts.  usage: run_engine.py [--tokens 120] [--only name,name]"""
import argparse
from common import *
ap = argparse.ArgumentParser(); ap.add_argument("--tokens", type=int, default=120); ap.add_argument("--only", default=""); a = ap.parse_args()
names = [n for n in prompt_names() if not a.only or n in a.only.split(",")]
print(f"{'prompt':16s} {'tokens':>6s} {'prefill s':>10s} {'decode tok/s':>13s}")
for n in names:
    r = run_engine(n, a.tokens if n not in ("document_6800", "repeat_1800") else 60); print(f"{n:16s} {r['prompt']:6d} {r['prefill_s']:10.2f} {r['decode_tok_s']:13.2f}", flush=True)
