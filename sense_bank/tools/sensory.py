import os, json, random, csv
from typing import Dict, List, Optional

from dotenv import load_dotenv
load_dotenv()

# -------- Data & memory --------

def load_corpus(data_dir: str) -> List[Dict]:
    rows: List[Dict] = []
    required = ["term","category","locale","era","weather","register","notes"]
    for name in os.listdir(data_dir):
        if not name.endswith(".csv"): continue
        path = os.path.join(data_dir, name)
        with open(path, "r", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                for k in required:
                    if k not in row:
                        raise RuntimeError(f"Missing column {k} in {name}")
                row["notes"] = row.get("notes") or ""
                for k in required:
                    row[k] = str(row[k]).strip()
                rows.append(row)
    if not rows:
        raise RuntimeError("No CSVs found in data/")
    return rows

def load_memory(mem_dir: str, key: Optional[str]) -> Dict:
    if not key: return {}
    os.makedirs(mem_dir, exist_ok=True)
    path = os.path.join(mem_dir, f"{key}.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_memory(mem_dir: str, key: str, data: Dict):
    os.makedirs(mem_dir, exist_ok=True)
    path = os.path.join(mem_dir, f"{key}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# -------- Suggestion logic --------

def _score(row, locale, era, weather, register):
    score = 0.0
    if row["locale"].lower() == locale.lower(): score += 3
    if row["era"].lower() == era.lower(): score += 2
    if weather.lower() == "any":
        score += 0.5
    elif row["weather"].lower() == weather.lower():
        score += 2
    if row["register"].lower() == register.lower(): score += 1.5
    return score

def suggest_sensory(
    rows: List[Dict],
    locale: str,
    era: str,
    weather: str = "any",
    register: str = "common",
    n: int = 6,
    memory: Dict = None
) -> List[Dict]:
    memory = memory or {}
    used = set(memory.get("used_terms", []))

    # Strict locale/era first
    strict = [r for r in rows if r["locale"].lower()==locale.lower() and r["era"].lower()==era.lower()]

    def soft_filter(frame: List[Dict]) -> List[Dict]:
        out = frame
        if weather.lower() != "any":
            out = [r for r in out if r["weather"].lower()==weather.lower()]
        if register:
            out = [r for r in out if r["register"].lower()==register.lower()]
        return out

    candidates = soft_filter(strict)
    if len(candidates) < n:
        candidates = strict[:]

    # Last resort: cross-locale ranked fallback
    if len(candidates) < n:
        ranked = []
        for r in rows:
            r = dict(r)
            r["__score"] = _score(r, locale, era, weather, register)
            ranked.append(r)
        ranked.sort(key=lambda x: x["__score"], reverse=True)
        seen = set(); merged = []
        for r in candidates + ranked:
            key = (r["term"], r["locale"], r["era"])
            if key not in seen:
                seen.add(key); merged.append(r)
        candidates = merged

    # Avoid repeats
    candidates = [r for r in candidates if r["term"] not in used]

    # Diversify categories
    order = ["smell","tactile","sound","visual","taste"]
    pool = []
    for cat in order:
        pool.extend([r for r in candidates if r["category"] == cat])

    random.shuffle(pool)
    selection = pool[:n]

    return [{
        "term": r["term"],
        "category": r["category"],
        "notes": r.get("notes",""),
        "locale": r["locale"],
        "era": r["era"],
        "weather": r["weather"],
        "register": r["register"]
    } for r in selection]

# -------- Rewrite using OpenAI-compatible API --------

def _openai_client():
    try:
        from openai import OpenAI
    except Exception as e:
        return None, str(e)
    base = os.environ.get("OPENAI_API_BASE") or os.environ.get("OPENAI_BASE_URL")
    key  = os.environ.get("OPENAI_API_KEY")
    model = os.environ.get("SENSEBANK_MODEL", "llama3.1")
    if not base or not key:
        return None, "Set OPENAI_API_BASE and OPENAI_API_KEY to enable rewriting."
    client = OpenAI(base_url=base, api_key=key)
    return (client, model), None

REWRITE_SYSTEM = """You are a historical fiction line editor. Convert 'tell' into vivid 'show' using the provided sensory cues.
Rules:
- Keep facts and time/place consistent.
- Use 2–5 of the provided cues naturally; don't list them.
- Avoid modern anachronisms.
- Keep it under 120 words unless the input is longer.
- Keep the original voice and POV.
"""

def rewrite_with_sensory(text: str, suggestions: List[dict]) -> Dict:
    client_model, err = _openai_client()
    if err or not client_model:
        return {"rewrite": None, "reasoning": f"LLM not configured: {err}", "model": None}
    client, model = client_model
    cues = "; ".join([f"{x['category']}: {x['term']} ({x.get('notes','')})" for x in suggestions])
    user = f"Rewrite the passage with concrete sensory detail:\n\nPASSAGE:\n{text}\n\nCUES:\n{cues}\n"
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role":"system","content": REWRITE_SYSTEM},
                      {"role":"user","content": user}],
            temperature=0.8,
        )
        out = resp.choices[0].message.content
        return {"rewrite": out, "model": model}
    except Exception as e:
        return {"rewrite": None, "reasoning": f"Model call failed: {e}", "model": model}
