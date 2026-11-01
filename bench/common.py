"""Shared helpers: paths, the engine's recommended environment, and a one-shot engine runner."""
import os, pathlib, subprocess
ROOT = pathlib.Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"; PROMPTS = ROOT / "bench" / "prompts"
CKPT = MODELS / "gemma-4-E2B-it-qat"; INDEX = MODELS / "engine_index.tsv"
DRAFTER = MODELS / "gemma-4-E2B-it-assistant"; DRAFTER_INDEX = MODELS / "drafter_index.tsv"
ENGINE = ROOT / "bin" / "gemma4_engine"

def cpus(n=8):   # one logical CPU per physical core on a 2-way SMT machine: 0,2,4,...  (edit for your CPU)
    return ",".join(str(2 * i) for i in range(n))

def engine_env(**extra):
    e = dict(os.environ, ENGINE_SERVE="1", ENGINE_CTX="8192", ENGINE_K="3", ENGINE_NGRAM="2", ENGINE_NGLEN="3", ENGINE_HEAD_SCREEN="1", ENGINE_PREFILL_INT8="1",
             ENGINE_ADAPTIVE_THREADS="1", ENGINE_THREAD_CUTOFF="1024", ENGINE_CPUS=cpus(), ENGINE_DRAFTER_DIR=str(DRAFTER), ENGINE_DRAFTER_INDEX=str(DRAFTER_INDEX))
    e.update({k: str(v) for k, v in extra.items()}); return e

def prompt_ids(name): return (PROMPTS / f"{name}.ids").read_text().strip()
def prompt_names(): return sorted(p.stem for p in PROMPTS.glob("*.ids"))

def run_engine(name, new_tokens, binary=ENGINE, env=None, timeout=1800):
    """Start the engine, send one empty request (warm-up / cold prefill) then one measured request; return its TIMING fields and generated ids."""
    ids = prompt_ids(name)
    out = subprocess.run([str(binary), str(CKPT), str(INDEX), "/dev/null", "0"], cwd=str(ROOT), env=env or engine_env(), input=f"0 {ids}\n{new_tokens} {ids}\n".encode(),
                         capture_output=True, timeout=timeout).stdout.decode().splitlines()
    tim = [l for l in out if l.startswith("TIMING ")]
    cold = dict(kv.split("=", 1) for kv in tim[0].split()[1:] if "=" in kv); warm = dict(kv.split("=", 1) for kv in tim[-1].split()[1:] if "=" in kv)
    gen = [l for l in out if l.startswith("GEN:")][-1].split()[1:]
    return {"prefill_s": float(cold["prefill_s"]), "decode_tok_s": float(warm["decode_tok_s"]), "tokens": gen, "prompt": int(warm["prompt"])}
