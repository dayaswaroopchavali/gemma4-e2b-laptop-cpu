#!/usr/bin/env python
"""OpenAI-compatible server in front of the Gemma-4 E2B CPU engine.

Same HTTP surface and `serve` command-line shape as `vllm serve`, so a client only changes its base URL:
    server.py serve <model dir> --port 8000 [--served-model-name NAME] [--max-model-len N] [--api-key KEY]
Unknown vLLM flags are accepted and ignored with a warning. Decoding is greedy (the engine certifies the argmax);
sampling parameters are accepted and ignored. One request runs at a time (the engine is single-stream);
others queue first-in first-out, and a shared prompt prefix is reused from the engine's KV cache.
"""
import argparse, asyncio, json, os, queue, subprocess, sys, threading, time, uuid
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                          # repository root: models/ and bin/ live here
VERSION = "0.1.0-cpu-gemma4"
STOP_IDS = {1, 106}                         # the ids the engine treats as end of sequence


class ApiError(Exception):
    def __init__(self, message, status=400, etype="invalid_request_error", param=None, code=None):
        self.message, self.status, self.etype, self.param, self.code = message, status, etype, param, code


def err_json(e: ApiError):
    return JSONResponse({"error": {"message": e.message, "type": e.etype, "param": e.param, "code": e.code or e.status}}, status_code=e.status)


class Engine:
    """The engine subprocess plus one worker thread that runs queued requests one at a time."""

    def __init__(self, binary, ckpt, index, env, log_path):
        self.jobs = queue.Queue(); self.alive = False; self.running = 0; self.dead_reason = ""
        self.log = open(log_path, "ab")
        everyone = set(range(os.cpu_count()))      # the server may be confined to spare cores; the engine pins its own threads, so give it the full mask
        self.p = subprocess.Popen([binary, ckpt, index, "/dev/null", "0"], cwd=str(ROOT), env=env, stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, stderr=self.log, text=True, bufsize=1, preexec_fn=lambda: os.sched_setaffinity(0, everyone))
        self.t = threading.Thread(target=self._worker, daemon=True); self.t.start()

    def _read_until_timing(self, emit):
        gen = None
        while True:
            line = self.p.stdout.readline()
            if line == "":
                raise RuntimeError("engine process exited")
            if line.startswith("FIRST "):
                emit(("tok", [int(line.split()[1])]))
            elif line.startswith("TOK"):
                emit(("tok", [int(x) for x in line.split()[1:]]))
            elif line.startswith("GEN:"):
                gen = [int(x) for x in line.split()[1:]]
            elif line.startswith("TIMING "):
                t = dict(kv.split("=", 1) for kv in line.split()[1:] if "=" in kv)
                return gen or [], t

    def _worker(self):
        try:
            self.p.stdin.write("0 2\n"); self.p.stdin.flush()      # handshake: empty request with just <bos>
            self._read_until_timing(lambda e: None); self.alive = True
            while True:
                job = self.jobs.get()
                if job is None:
                    return
                ids, max_new, loop, out = job
                emit = lambda ev: loop.call_soon_threadsafe(out.put_nowait, ev)
                self.running = 1
                try:
                    self.p.stdin.write(f"{max_new} {' '.join(map(str, ids))}\n"); self.p.stdin.flush()
                    gen, timing = self._read_until_timing(emit); emit(("done", gen, timing))
                except Exception as ex:
                    emit(("error", str(ex))); raise
                finally:
                    self.running = 0
        except Exception as ex:                                       # engine died: fail the current and every queued request
            self.alive = False; self.dead_reason = str(ex)
            while True:
                try:
                    job = self.jobs.get_nowait()
                except queue.Empty:
                    break
                if job: job[2].call_soon_threadsafe(job[3].put_nowait, ("error", str(ex)))

    def submit(self, ids, max_new):
        loop = asyncio.get_running_loop(); out = asyncio.Queue(); self.jobs.put((ids, max_new, loop, out))
        if not self.alive and self.dead_reason:
            out.put_nowait(("error", self.dead_reason))
        return out

    def close(self):
        self.jobs.put(None)
        try: self.p.stdin.close()
        except Exception: pass
        self.p.terminate()


class Detok:
    """Incremental detokenizer in vLLM's prefix/read-offset scheme (decodes only a short window per token, never the whole text);
    holds back text that could still become a stop string and never emits half a character."""

    def __init__(self, tok, stops):
        self.tok, self.ids, self.stops = tok, [], stops
        self.prefix_off = self.read_off = 0; self.text = ""; self.sent = 0
        self.hold = max((len(s) for s in stops), default=1) - 1; self.stopped = False

    def _advance(self, final):
        prefix = self.tok.decode(self.ids[self.prefix_off:self.read_off], skip_special_tokens=True)
        full = self.tok.decode(self.ids[self.prefix_off:], skip_special_tokens=True)
        if len(full) > len(prefix) and (final or not full.endswith("\ufffd")):
            self.text += full[len(prefix):]; self.prefix_off, self.read_off = self.read_off, len(self.ids)

    def push(self, new_ids, final=False):
        if self.stopped: return ""
        self.ids.extend(new_ids); self._advance(final)
        text = self.text
        for s in self.stops:
            i = text.find(s, max(0, self.sent - len(s)))
            if i >= 0: self.stopped = True; text = text[:i]; break
        safe = len(text) if (final or self.stopped) else max(self.sent, len(text) - self.hold)
        d = text[self.sent:safe]; self.sent = safe; return d


class State:
    expose_timing = False; engine: Engine = None; tok = None; model = ""; max_len = 8192; api_key = None
    counters = dict(prompt=0, gen=0, ok_stop=0, ok_length=0, started=0)


S = State()
app = FastAPI(title="Gemma-4 CPU engine (OpenAI-compatible)", version=VERSION)


@app.exception_handler(ApiError)
async def _api_error(_, e: ApiError): return err_json(e)


@app.middleware("http")
async def _auth(request: Request, call_next):
    if S.api_key and request.url.path.startswith("/v1") and request.headers.get("authorization") != f"Bearer {S.api_key}":
        return err_json(ApiError("Invalid API key", 401, "authentication_error"))
    return await call_next(request)


def check_ready():
    if S.engine is None or not S.engine.alive:
        raise ApiError("engine not ready" if S.engine and not S.engine.dead_reason else f"engine unavailable: {S.engine.dead_reason}", 503, "server_error")


def stops_of(body):
    st = body.get("stop"); st = [] if st is None else ([st] if isinstance(st, str) else list(st))
    if len(st) > 4: raise ApiError("stop accepts at most 4 sequences", param="stop")
    return [s for s in st if s]


def common_checks(body):
    if int(body.get("n", 1) or 1) != 1: raise ApiError("n > 1 is not supported by this engine", param="n")
    if body.get("logprobs") or body.get("top_logprobs"): raise ApiError("logprobs are not supported: the engine certifies the argmax without materialising logits", param="logprobs")


def message_text(m):
    c = m.get("content")
    if c is None: return ""
    if isinstance(c, str): return c
    out = []
    for part in c:
        t = part.get("type")
        if t == "text": out.append(part.get("text", ""))
        elif t in ("input_audio", "audio_url", "image_url", "video_url"):
            raise ApiError(f"content part '{t}' is not supported yet by the CPU engine server (text only)", param="messages")
        else: raise ApiError(f"unsupported content part type '{t}'", param="messages")
    return "".join(out)


def build_chat_ids(body):
    msgs = body.get("messages")
    if not isinstance(msgs, list) or not msgs: raise ApiError("'messages' must be a non-empty list", param="messages")
    norm = []
    for m in msgs:
        d = dict(m); d["content"] = message_text(m); norm.append(d)
    kw = dict(body.get("chat_template_kwargs") or {})
    if body.get("tools"): kw["tools"] = body["tools"]
    try:
        text = S.tok.apply_chat_template(norm, tokenize=False, add_generation_prompt=body.get("add_generation_prompt", True), **kw)
    except Exception as ex:
        raise ApiError(f"chat template failed: {ex}", param="messages")
    return S.tok.encode(text, add_special_tokens=False)


def plan(ids, body):
    mt = body.get("max_completion_tokens", body.get("max_tokens"))
    if len(ids) >= S.max_len: raise ApiError(f"This model's maximum context length is {S.max_len} tokens. However, you requested {len(ids)} prompt tokens.", param="messages", code="context_length_exceeded")
    room = S.max_len - len(ids)
    if mt is None: mt = room
    mt = int(mt)
    if mt < 1: raise ApiError("max_tokens must be at least 1", param="max_tokens")
    if mt > room: raise ApiError(f"This model's maximum context length is {S.max_len} tokens. However, you requested {mt + len(ids)} tokens ({len(ids)} in the prompt, {mt} in the completion).", param="max_tokens", code="context_length_exceeded")
    return mt


async def run(ids, max_new, stops):
    """Yield ('delta', text) per update, then ('end', finish_reason, completion_tokens, timing)."""
    check_ready(); S.counters["started"] += 1; S.counters["prompt"] += len(ids)
    out = S.engine.submit(ids, max_new); dt = Detok(S.tok, stops); n = 0; stop_hit = False
    while True:
        ev = await out.get()
        if ev[0] == "error": raise ApiError(f"engine error: {ev[1]}", 500, "server_error")
        if ev[0] == "tok":
            n += len(ev[1])
            if not stop_hit:
                d = dt.push(ev[1]); stop_hit = dt.stopped
                if d: yield ("delta", d)
        else:
            _, gen, timing = ev
            if not stop_hit:
                d = dt.push([], final=True); stop_hit = dt.stopped
                if d: yield ("delta", d)
            ended_on_eos = bool(gen) and gen[-1] in STOP_IDS
            fr = "stop" if (stop_hit or ended_on_eos) else "length"
            S.counters["gen"] += len(gen); S.counters["ok_stop" if fr == "stop" else "ok_length"] += 1
            yield ("end", fr, len(gen), int(timing.get("reused", 0)), timing); return


def usage(pt, ct, cached=0, tm=None):
    u = {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct, "prompt_tokens_details": {"cached_tokens": cached}}
    if S.expose_timing and tm: u["engine_timing"] = tm
    return u


def sse(obj): return f"data: {json.dumps(obj, separators=(',', ':'))}\n\n"


async def serve_request(body, ids, kind):
    max_new = plan(ids, body); stops = stops_of(body); stream = bool(body.get("stream"))
    rid = ("chatcmpl-" if kind == "chat" else "cmpl-") + uuid.uuid4().hex; created = int(time.time()); model = body.get("model") or S.model
    headers = {"X-Engine-Decoding": "greedy"}
    if not stream:
        text = []; fr = "stop"; ct = 0; cached = 0; tm = None
        async for ev in run(ids, max_new, stops):
            if ev[0] == "delta": text.append(ev[1])
            else: _, fr, ct, cached, tm = ev
        t = "".join(text)
        ch = {"index": 0, "message": {"role": "assistant", "content": t}, "finish_reason": fr, "logprobs": None} if kind == "chat" else {"index": 0, "text": t, "logprobs": None, "finish_reason": fr}
        return JSONResponse({"id": rid, "object": "chat.completion" if kind == "chat" else "text_completion", "created": created, "model": model, "choices": [ch], "usage": usage(len(ids), ct, cached, tm)}, headers=headers)
    include_usage = bool((body.get("stream_options") or {}).get("include_usage"))
    obj = "chat.completion.chunk" if kind == "chat" else "text_completion"

    def chunk(delta=None, text=None, fr=None):
        c = {"index": 0, "finish_reason": fr}
        if kind == "chat": c["delta"] = delta if delta is not None else {}
        else: c["text"] = text or ""; c["logprobs"] = None
        return {"id": rid, "object": obj, "created": created, "model": model, "choices": [c]}

    async def gen():
        try:
            if kind == "chat": yield sse(chunk({"role": "assistant", "content": ""}))
            ct = 0; fr = "stop"; cached = 0; tm = None
            async for ev in run(ids, max_new, stops):
                if ev[0] == "delta": yield sse(chunk({"content": ev[1]}) if kind == "chat" else chunk(text=ev[1]))
                else: _, fr, ct, cached, tm = ev
            yield sse(chunk(fr=fr))
            if include_usage: yield sse({"id": rid, "object": obj, "created": created, "model": model, "choices": [], "usage": usage(len(ids), ct, cached, tm)})
        except ApiError as e:
            yield sse({"error": {"message": e.message, "type": e.etype, "code": e.status}})
        yield "data: [DONE]\n\n"
    return StreamingResponse(gen(), media_type="text/event-stream", headers={**headers, "Cache-Control": "no-cache"})


async def json_body(request):
    try: b = await request.json()
    except Exception: raise ApiError("request body is not valid JSON")
    if not isinstance(b, dict): raise ApiError("request body must be a JSON object")
    return b


@app.post("/v1/chat/completions")
async def chat(request: Request):
    body = await json_body(request); common_checks(body); check_ready()
    return await serve_request(body, build_chat_ids(body), "chat")


@app.post("/v1/completions")
async def completions(request: Request):
    body = await json_body(request); common_checks(body); check_ready()
    p = body.get("prompt")
    if isinstance(p, list) and p and isinstance(p[0], list): p = p[0]
    if isinstance(p, list) and p and isinstance(p[0], str): p = p[0]
    if isinstance(p, str): ids = S.tok.encode(p, add_special_tokens=True)
    elif isinstance(p, list) and all(isinstance(x, int) for x in p) and p: ids = list(p)
    else: raise ApiError("'prompt' must be a string or a list of token ids", param="prompt")
    return await serve_request(body, ids, "completion")


@app.get("/v1/models")
async def models():
    return {"object": "list", "data": [{"id": S.model, "object": "model", "created": int(time.time()), "owned_by": "vllm", "root": S.model, "parent": None, "max_model_len": S.max_len}]}


@app.get("/health")
async def health(): return Response(status_code=200 if S.engine and S.engine.alive else 503)


@app.get("/version")
async def version(): return {"version": VERSION}


@app.post("/tokenize")
async def tokenize(request: Request):
    b = await json_body(request)
    ids = build_chat_ids(b) if "messages" in b else S.tok.encode(b.get("prompt", ""), add_special_tokens=b.get("add_special_tokens", True))
    return {"count": len(ids), "max_model_len": S.max_len, "tokens": ids}


@app.post("/detokenize")
async def detokenize(request: Request):
    b = await json_body(request); return {"prompt": S.tok.decode(b.get("tokens", []), skip_special_tokens=False)}


@app.get("/metrics")
async def metrics():
    c = S.counters; q = S.engine.jobs.qsize() if S.engine else 0
    lines = [f'vllm:num_requests_running {S.engine.running if S.engine else 0}', f'vllm:num_requests_waiting {q}', f'vllm:prompt_tokens_total{{model_name="{S.model}"}} {c["prompt"]}',
             f'vllm:generation_tokens_total{{model_name="{S.model}"}} {c["gen"]}', f'vllm:request_success_total{{model_name="{S.model}",finished_reason="stop"}} {c["ok_stop"]}',
             f'vllm:request_success_total{{model_name="{S.model}",finished_reason="length"}} {c["ok_length"]}']
    return Response("\n".join(lines) + "\n", media_type="text/plain; version=0.0.4")


def parse_args(argv):
    if argv and argv[0] == "serve": argv = argv[1:]
    model_pos = None
    if argv and not argv[0].startswith("-"): model_pos, argv = argv[0], argv[1:]      # `vllm serve MODEL ...`: only the first argument is the model
    ap = argparse.ArgumentParser(prog="server.py serve", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=None, help="model directory (tokenizer + chat template + weights); or pass it first, as with vllm serve")
    ap.add_argument("--host", default="0.0.0.0"); ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--served-model-name", nargs="+", default=None); ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--engine-bin", default=str(ROOT / "bin" / "gemma4_engine")); ap.add_argument("--engine-index", default=str(ROOT / "models" / "engine_index.tsv"))
    ap.add_argument("--drafter-dir", default=str(ROOT / "models" / "gemma-4-E2B-it-assistant")); ap.add_argument("--drafter-index", default=str(ROOT / "models" / "drafter_index.tsv"))
    ap.add_argument("--engine-cpus", default="0,2,4,6,8,10,12,14", help="physical cores the engine threads are pinned to")
    ap.add_argument("--server-cpus", default="auto", help="cores for the HTTP server itself: 'auto' = every core the engine does not use, 'all' = no restriction")
    ap.add_argument("--slots", type=int, default=4, help="conversations whose KV cache is kept (~150 MB each at 8k context); a request reuses the slot sharing the longest prefix")
    ap.add_argument("--no-mlock", dest="mlock", action="store_false", help="do not lock the engine's anonymous memory (needs RLIMIT_MEMLOCK; locks as much as allowed)")
    ap.add_argument("--expose-timing", action="store_true", help="add the engine's own timing to the usage block (diagnostics)")
    ap.add_argument("--engine-log", default="/tmp/gemma4_engine.log"); ap.add_argument("--exact", action="store_true", help="exact float decode (default: int8 decode, ~12-25%% faster)")
    a, extra = ap.parse_known_args(argv); a.model = model_pos or a.model or str(ROOT / "models" / "gemma-4-E2B-it-qat"); return a, extra


def engine_env(a):
    e = dict(os.environ, ENGINE_SERVE="1", ENGINE_STREAM="1", ENGINE_CTX=str(a.max_model_len), ENGINE_K="3", ENGINE_NGRAM="2", ENGINE_NGLEN="3", ENGINE_HEAD_SCREEN="1",
             ENGINE_PREFILL_INT8="1", ENGINE_CPUS=a.engine_cpus, ENGINE_ADAPTIVE_THREADS="1", ENGINE_THREAD_CUTOFF="1024", ENGINE_DRAFTER_DIR=a.drafter_dir, ENGINE_DRAFTER_INDEX=a.drafter_index,
             ENGINE_SLOTS=str(a.slots), ENGINE_KEEPWARM="1")
    if a.mlock: e["ENGINE_MLOCK"] = "1"
    if a.exact: e["ENGINE_DECODE_INT8"] = "0"
    return e


def main():
    a, extra = parse_args(sys.argv[1:])
    if a.server_cpus != "all":      # keep Python off the engine's pinned, spin-waiting cores (a preempted worker stalls every barrier)
        eng = {int(x) for x in a.engine_cpus.split(",")}; spare = set(range(os.cpu_count())) - eng if a.server_cpus == "auto" else {int(x) for x in a.server_cpus.split(",")}
        if spare: os.sched_setaffinity(0, spare)
    if extra: print(f"[engine-server] ignoring vLLM flags this engine does not use: {' '.join(extra)}", file=sys.stderr)
    if not Path(a.engine_bin).exists(): sys.exit(f"engine binary not found: {a.engine_bin}\nbuild it with: {ROOT}/build.sh")
    S.expose_timing = a.expose_timing; S.model = (a.served_model_name or [a.model])[0]; S.max_len = a.max_model_len; S.api_key = a.api_key
    S.tok = AutoTokenizer.from_pretrained(a.model)

    @app.on_event("startup")
    async def _start():
        S.engine = Engine(a.engine_bin, a.model, a.engine_index, engine_env(a), a.engine_log)
        t0 = time.time()
        while not S.engine.alive and not S.engine.dead_reason and time.time() - t0 < 300: await asyncio.sleep(0.2)
        print(f"[engine-server] engine ready in {time.time() - t0:.1f}s; serving '{S.model}' on {a.host}:{a.port}", file=sys.stderr)

    @app.on_event("shutdown")
    async def _stop():
        if S.engine: S.engine.close()
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
