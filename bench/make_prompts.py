"""Write the neutral benchmark prompts to bench/prompts/*.txt and *.ids (token ids after the chat template).
Lengths are tuned with the model tokenizer to be close to the cases in docs/RESULTS.md (about 20, 30, 50, 70, 150, 600, 1,900 and 6,800 tokens).
usage: make_prompts.py [model_dir]   (default models/gemma-4-E2B-it-qat)"""
import sys, pathlib
from transformers import AutoTokenizer
ROOT = pathlib.Path(__file__).resolve().parent.parent; OUT = ROOT / "bench" / "prompts"; OUT.mkdir(parents=True, exist_ok=True)
tok = AutoTokenizer.from_pretrained(sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "models" / "gemma-4-E2B-it-qat"))
chat = lambda text: tok.encode(tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True), add_special_tokens=False)
fixed = {
  "short_command": "Set a timer for ten minutes and name it tea.",
  "call_request": "Call my brother and ask whether he can collect the parcel from the post office before six this evening.",
  "alert_text": "Alert: the temperature sensor in cold room 3 has read 9 degrees for the last 15 minutes, above the 4 degree limit. Write a two sentence notice for the on-call technician.",
  "tool_json": 'You can call tools. Reply ONLY with a JSON object of the form {"tool": name, "args": {...}}. Tools: send_message(chat, text), set_timer(minutes, label), search_notes(query). Request: tell Priya that the delivery has moved to Thursday afternoon.',
  "message_summary": "Summarise these six unread messages in three short bullet points.\n[09:02] Ana: Can we move the planning call to 3pm?\n[09:05] Ben: 3pm works for me, I will share the notes before.\n[09:11] Ana: Great. Also the venue booking for Friday needs a deposit.\n[09:14] Chris: I can pay the deposit today, send me the account details.\n[09:20] Ana: Sent by email. Please confirm when it is done.\n[09:31] Chris: Deposit paid, receipt attached.",
}
def pad(base, line_fn, target):          # grow a text line by line until it reaches the target token count
    lines, i = [], 0
    while len(chat("\n".join(lines))) < target: lines.append(line_fn(i)); i += 1
    return "\n".join(lines)
times = ["09:%02d" % (i % 60) for i in range(2000)]; who = ["Ana", "Ben", "Chris", "Dina", "Eli"]; what = ["confirmed the schedule for the next sprint review", "asked for the latest figures from the finance sheet", "shared the draft of the customer reply for approval", "reported that the staging build passed all checks", "moved the supplier call to the afternoon slot", "flagged a delay on the parts order from the warehouse"]
fixed["running_summary"] = "Here is the running summary so far, followed by new messages. Update the summary in under 80 words.\n" + pad("", lambda i: f"[{times[i]}] {who[i % 5]} {what[i % 6]} (item {i})", 581 - 12)
base = "The warehouse reported that truck seven is on its way to the loading bay while truck twelve waits at the depot and asks for a status update from the control room. "
fixed["repeat_1800"] = None
docs = "\n\n".join(p.read_text() for p in sorted((ROOT / "docs").glob("*.md")) if p.name != "RESULTS.md") + "\n" + (ROOT / "README.md").read_text()
for name, text in fixed.items():
    if name == "repeat_1800": text = "Summarise in one line:\n" + base * 60 + "\nSummary:"; ids = tok.encode(text, add_special_tokens=True)           # raw, no chat template: highly repetitive text
    else: ids = chat(text)
    (OUT / f"{name}.txt").write_text(text); (OUT / f"{name}.ids").write_text(" ".join(map(str, ids))); print(f"{name:16s} {len(ids):5d} tokens")
long = "Read the following technical notes, then explain the three most important trade-offs in a concise answer.\n\n"
while len(chat(long)) < 6800: long += docs[len(long) % len(docs):][:400] + "\n"
ids = chat(long); (OUT / "document_6800.txt").write_text(long); (OUT / "document_6800.ids").write_text(" ".join(map(str, ids))); print(f"document_6800    {len(ids):5d} tokens")
