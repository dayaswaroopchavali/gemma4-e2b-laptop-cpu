"""Black-box checks against a running server with the official OpenAI client.  usage: test_server.py [base_url]   (run bench/make_prompts.py first)"""
import sys, time, json, threading, urllib.request, pathlib
from openai import OpenAI, BadRequestError
ROOT = pathlib.Path(__file__).resolve().parent.parent; P = ROOT / "bench" / "prompts"
base = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
c = OpenAI(base_url=base + "/v1", api_key="EMPTY"); M = c.models.list().data[0].id; ok = fail = 0
def check(name, cond, info=""):
    global ok, fail; ok += bool(cond); fail += (not cond); print(("PASS " if cond else "FAIL ") + name + (f"  [{info}]" if info and not cond else ""), flush=True)
txt = lambda n: (P / f"{n}.txt").read_text()
check("models", c.models.list().data[0].max_model_len > 0)
for name in ["alert_text", "call_request", "tool_json"]:
    msgs = [{"role": "user", "content": txt(name)}]
    r = c.chat.completions.create(model=M, messages=msgs, max_tokens=120, temperature=0); full = r.choices[0].message.content
    check(f"chat {name}: finished normally with usage", r.choices[0].finish_reason == "stop" and r.usage.completion_tokens > 1 and r.usage.prompt_tokens > 10 and len(full) > 0, str(r.usage))
    t0 = time.time(); first = None; parts = []; fin = None; role = None; u = None
    for ch in c.chat.completions.create(model=M, messages=msgs, max_tokens=120, temperature=0, stream=True, stream_options={"include_usage": True}):
        if ch.choices:
            d = ch.choices[0].delta; role = d.role or role
            if d.content: first = first or time.time() - t0; parts.append(d.content)
            fin = ch.choices[0].finish_reason or fin
        else: u = ch.usage
    check(f"stream {name}: text identical to non-stream, role, finish, usage", "".join(parts) == full and role == "assistant" and fin == "stop" and u.completion_tokens == r.usage.completion_tokens, "".join(parts)[:60])
    print(f"     time to first content chunk {first*1000:.0f} ms, {len(parts)} chunks")
r = c.chat.completions.create(model=M, messages=[{"role": "user", "content": "Count from one to ten in words."}], max_tokens=7)
check("max_tokens=7 -> finish_reason length", r.choices[0].finish_reason == "length" and r.usage.completion_tokens == 7)
r = c.chat.completions.create(model=M, messages=[{"role": "user", "content": "Count from one to ten in words."}], max_tokens=60, stop=["Five"])
check("stop string cuts the text", "Five" not in r.choices[0].message.content and r.choices[0].finish_reason == "stop", r.choices[0].message.content)
r = c.completions.create(model=M, prompt="The capital of France is", max_tokens=8)
s = "".join(x.choices[0].text for x in c.completions.create(model=M, prompt="The capital of France is", max_tokens=8, stream=True))
check("/v1/completions works and streaming == non-streaming", r.object == "text_completion" and len(r.choices[0].text) > 0 and s == r.choices[0].text, s)
r2 = c.chat.completions.create(model=M, messages=[{"role": "system", "content": "Answer in one word."}, {"role": "user", "content": "Capital of Japan?"}, {"role": "assistant", "content": "Tokyo"}, {"role": "user", "content": "And of Italy?"}], max_tokens=20, temperature=0.9, top_p=0.5, seed=3)
check("multi-turn; sampling parameters accepted (decoding is greedy)", "Rome" in r2.choices[0].message.content, r2.choices[0].message.content)
for label, kw in [("n=2", dict(n=2)), ("logprobs", dict(logprobs=True)), ("audio part", None), ("too long", dict(max_tokens=99999))]:
    try:
        if kw is None: c.chat.completions.create(model=M, messages=[{"role": "user", "content": [{"type": "input_audio", "input_audio": {"data": "AAAA", "format": "wav"}}]}])
        else: c.chat.completions.create(model=M, messages=[{"role": "user", "content": "hi"}], **kw)
        check(f"error {label}", False, "no error")
    except BadRequestError as e: check(f"error {label} -> 400 in OpenAI format", e.status_code == 400 and bool(e.message), str(e)[:80])
def raw(path, body=None):
    req = urllib.request.Request(base + path, json.dumps(body).encode() if body else None, {"Content-Type": "application/json"}); return urllib.request.urlopen(req).read().decode()
check("version and metrics endpoints", "version" in raw("/version") and "vllm:num_requests_running" in raw("/metrics"))
tk = json.loads(raw("/tokenize", {"messages": [{"role": "user", "content": "hello"}]}))
check("tokenize + detokenize", tk["count"] == len(tk["tokens"]) and "hello" in json.loads(raw("/detokenize", {"tokens": tk["tokens"]}))["prompt"])
res = {}
def one(i): res[i] = c.chat.completions.create(model=M, messages=[{"role": "user", "content": f"Reply with the number {i} only."}], max_tokens=10).choices[0].message.content
th = [threading.Thread(target=one, args=(i,)) for i in range(4)]; [t.start() for t in th]; [t.join() for t in th]
check("4 simultaneous requests are all served (queued)", all(str(i) in res[i] for i in range(4)), str(res))
def ask(text):
    t0 = time.time(); first = None; u = None
    for ch in c.chat.completions.create(model=M, messages=[{"role": "user", "content": text}], max_tokens=8, stream=True, stream_options={"include_usage": True}):
        if ch.choices and ch.choices[0].delta.content and first is None: first = time.time() - t0
        if not ch.choices: u = ch.usage
    return first, u
A, B, C = txt("running_summary") + " ", txt("message_summary") + " ", txt("alert_text") + " "
ask(A); ask(B); ask(C)                       # three different conversations, one after another
fa, ua = ask(A); fb, ub = ask(B)
check("alternating conversations: each returns with its prefix cached", ua.prompt_tokens_details.cached_tokens >= ua.prompt_tokens - 2 and ub.prompt_tokens_details.cached_tokens >= ub.prompt_tokens - 2, f"{ua} {ub}")
check("a cached conversation answers fast (first token < 0.3 s)", fa < 0.3 and fb < 0.3, f"{fa:.3f} {fb:.3f}")
print(f"\n{ok} passed, {fail} failed")
