#!/usr/bin/env python3
import os, sys, json, argparse, re
from pathlib import Path
from typing import List, Dict, Optional
import requests

# ---- env / defaults ----------------------------------------------------------
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "http://localhost:11434/v1")
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY", "ollama")
MODEL_ID        = os.getenv("SENSEBANK_MODEL", "llama3.1:8b")
MCP_URL         = os.getenv("MCP_URL", "http://127.0.0.1:8000/mcp/")

# ---- tiny OpenAI-compatible JSON call (works with Ollama) --------------------
def chat_json(system: str, user: str, model: str = MODEL_ID, temperature: float = 0.2) -> Dict:
    url = f"{OPENAI_API_BASE.rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    payload = {
        "model": model,
        "messages": [{"role":"system","content": system},{"role":"user","content": user}],
        # Many OpenAI-compatible endpoints ignore this, but it helps with some:
        "response_format": {"type": "json_object"},
        "temperature": temperature,
    }
    r = requests.post(url, headers=headers, json=payload, timeout=120)
    r.raise_for_status()
    content = r.json()["choices"][0]["message"]["content"]
    # be robust to ```json fences
    try:
        return json.loads(content)
    except Exception:
        m = re.search(r"\{.*}", content, flags=re.S)
        if m:
            return json.loads(m.group(0))
        raise ValueError(f"Model did not return JSON:\n{content}")

# ---- chunking ----------------------------------------------------------------
def read_text(path: Optional[str]) -> str:
    if not path or path == "-":
        return sys.stdin.read()
    return Path(path).read_text(encoding="utf-8")

def chunk_paragraphs(text: str, max_chars: int = 4000) -> List[str]:
    pars = re.split(r"\n\s*\n", text.strip())
    out, cur = [], ""
    for p in pars:
        if len(cur) + len(p) + 2 > max_chars and cur:
            out.append(cur.strip())
            cur = p
        else:
            cur = (cur + "\n\n" + p) if cur else p
    if cur.strip():
        out.append(cur.strip())
    return out

# ---- extraction prompt -------------------------------------------------------
SYSTEM = """You extract *explicit* sensory details from literary fiction passages.

Return strict JSON: {"items":[
  {"term": "...", "category": "smell|sound|taste|touch|sight", "notes": "..."},
  ...
]}

Rules:
- Only include cues *stated or strongly implied* by the passage; do not invent.
- "term" should be short, evocative, reusable (e.g., "inkstone slurry", "wet lacquer rail").
- "notes" adds a tight gloss (<= 12 words), not a full sentence.
- Use the five categories exactly: smell, sound, taste, touch, sight.
- No anachronisms. Use wording natural to the passage's tone.
- 0..12 items per chunk is fine; return fewer if the text is sparse.
"""

USER_TMPL = """Extract sensory items from this passage. Output JSON only.

PASSAGE:
{passage}
"""

def extract_items(chunk: str) -> List[Dict]:
    obj = chat_json(SYSTEM, USER_TMPL.format(passage=chunk))
    items = obj.get("items", []) if isinstance(obj, dict) else []
    cleaned = []
    for it in items:
        term = (it.get("term","") or "").strip()
        cat  = (it.get("category","") or "").strip().lower()
        notes= (it.get("notes","") or "").strip()
        if term and cat in {"smell","sound","taste","touch","sight"}:
            cleaned.append({"term": term, "category": cat, "notes": notes})
    return cleaned

# ---- MCP client (minimal) ----------------------------------------------------
# Tiny inline client for FastMCP streamable-http
import uuid, json as _json, requests as _requests

def mcp_call(tool: str, args: Dict) -> Dict:
    """POST to streamable-http MCP endpoint with a synthetic session."""
    sid = uuid.uuid4().hex
    # open session
    _requests.post(MCP_URL.rstrip("/"), allow_redirects=True, timeout=30)
    # list tools (optional: warms session)
    _requests.post(MCP_URL.rstrip("/"), json={"type":"ListToolsRequest","session_id":sid}, timeout=30)
    # call tool
    r = _requests.post(
        MCP_URL.rstrip("/"),
        json={"type":"CallToolRequest","session_id":sid,"name":tool,"arguments":args},
        timeout=120
    )
    r.raise_for_status()
    return r.json()

def sense_add(record: Dict, defaults: Dict) -> Dict:
    payload = {
        "term": record["term"],
        "category": record["category"],
        "locale": defaults["locale"],
        "era": defaults["era"],
        "weather": defaults["weather"],
        "register": defaults["register"],
        "notes": record.get("notes",""),
        "file": defaults.get("file")
    }
    return mcp_call("sense_add", payload)

# ---- main --------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Extract sensory details from fiction and add to Sense-Bank.")
    ap.add_argument("--file", help="Input .txt/.md (or '-' for stdin)", required=False)
    ap.add_argument("--locale", required=True)
    ap.add_argument("--era", required=True)
    ap.add_argument("--register", default="common")
    ap.add_argument("--weather",  default="any")
    ap.add_argument("--target-file", dest="target_file", help="Force CSV filename (optional)")
    ap.add_argument("--max-chars", type=int, default=4000)
    ap.add_argument("--dry-run", action="store_true", help="Do not write to CSVs")
    ap.add_argument("--print-json", action="store_true", help="Print extracted JSON")
    args = ap.parse_args()

    text = read_text(args.file)
    chunks = chunk_paragraphs(text, max_chars=args.max_chars)

    all_items: List[Dict] = []
    for i, ch in enumerate(chunks, 1):
        items = extract_items(ch)
        all_items.extend(items)

    # dedupe by (term, category)
    seen = set()
    uniq: List[Dict] = []
    for it in all_items:
        key = (it["term"].lower(), it["category"])
        if key not in seen:
            seen.add(key)
            uniq.append(it)

    if args.print_json:
        print(json.dumps({"items": uniq}, ensure_ascii=False, indent=2))

    defaults = {
        "locale": args.locale, "era": args.era,
        "register": args.register, "weather": args.weather
    }
    if args.target_file:
        defaults["file"] = Path(args.target_file).name

    if args.dry_run:
        print(f"[dry-run] would add {len(uniq)} items")
        for it in uniq:
            print(f"  - {it['category']}: {it['term']}  — {it.get('notes','')}")
        return

    # write via MCP
    added, exists = 0, 0
    for it in uniq:
        res = sense_add(it, defaults)
        status = res.get("status")
        if status == "added":
            added += 1
        elif status == "exists":
            exists += 1
        else:
            print("WARN:", res)
    print(f"Done. added={added} exists={exists} total={len(uniq)}")

if __name__ == "__main__":
    main()
