#!/usr/bin/env python3
import csv
from pathlib import Path
from typing import Protocol, runtime_checkable
import re

from dotenv import load_dotenv
from difflib import SequenceMatcher

from collections import defaultdict
from typing import Any, List, Tuple


load_dotenv()

from mcp.server.fastmcp import FastMCP
from typing import Optional, Dict, cast

@runtime_checkable
class SupportsWriteStr(Protocol):
    def write(self, s: str, /) -> int: ...

# Reuse Sense Bank functions
from sense_bank.tools.sensory import (
    load_corpus, suggest_sensory, rewrite_with_sensory,
    load_memory, save_memory,
)

ROOT = Path(__file__).parent
EXPORT_DIR = ROOT / "exports"  # add this near DATA_DIR/MEM_DIR
DATA_DIR = ROOT / "data"
MEM_DIR = ROOT / "memory"
HEADER = ["term", "category", "locale", "era", "weather", "register", "notes"]

mcp = FastMCP("sense-bank")

# ---------- helpers

DEFAULT_FILES = {
    "Japan": "jp_sensory.csv",
    "Andes": "andes_sensory.csv",
}

# near the top
def _json(obj: dict) -> dict:
    # explicit JSON content for MCP responses
    return {"type": "json", "json": obj}


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", (s or "all").strip().lower()).strip("_")

def _split_set(csv_vals: Optional[str]) -> set[str]:
    return {t.strip().lower() for t in (csv_vals or "").split(",") if t.strip()}

# Optional “defaults” for /validate if you want built-ins:
DEFAULT_ALLOWED_REGISTERS = {
    "court","common","ritual","market","monastic","maritime","festival","military","scholar","scholar","garden"
}
DEFAULT_ALLOWED_WEATHER = {
    "any","rain","summer","winter","monsoon","dry season","winter snow","dawn","night"
}
def _default_file_for_locale(locale: str) -> str:
    return DEFAULT_FILES.get(locale, f"{(locale or 'misc').lower()}_sensory.csv")

def _sim(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def _sanitize(s: Optional[str]) -> str:
    return " ".join((s or "").replace("\r", " ").replace("\n", " ").split())


def _row_key(d: Dict) -> Tuple[str, str, str, str]:
    return (
        (d.get("term", "")).strip().lower(),
        (d.get("category", "")).strip().lower(),
        (d.get("locale", "")).strip().lower(),
        (d.get("era", "")).strip().lower(),
    )


def _load_rows(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        return [{k: (v or "") for k, v in row.items()} for row in r]


def _write_rows(path: Path, rows: List[Dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(cast(SupportsWriteStr, f), fieldnames=HEADER)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in HEADER})
    tmp.replace(path)


def _find_in_any_csv(target_key: Tuple[str, str, str, str]) -> Optional[Path]:
    for p in DATA_DIR.glob("*.csv"):
        rows = _load_rows(p)
        for r in rows:
            if _row_key(r) == target_key:
                return p
    return None

def _md_split_row(line: str) -> List[str]:
    # Split a markdown table row into cells, handling escaped \| safely.
    cells, buf, esc = [], [], False
    # trim leading/trailing pipes
    s = line.strip()
    if s.startswith("|"): s = s[1:]
    if s.endswith("|"): s = s[:-1]
    for ch in s:
        if esc:
            buf.append(ch); esc = False; continue
        if ch == "\\":
            esc = True; continue
        if ch == "|":
            cells.append("".join(buf).strip()); buf = []
        else:
            buf.append(ch)
    cells.append("".join(buf).strip())
    return cells


# ---------- tools

@mcp.tool()
def sense_search(
    q: str,
    file: Optional[str] = None,
    locale: Optional[str] = None,
    era: Optional[str] = None,
    category: Optional[str] = None,
    top_k: int = 10,
    weight_term: float = 0.7,    # term vs notes mix
    weight_notes: float = 0.3
) -> Dict:
    """
    Fuzzy search over term+notes. Returns top_k with similarity score [0,1].
    """
    res = sense_list(locale=locale, era=era, category=category, file=file, limit=0)
    items = res.get("items", [])
    scored = []
    for r in items:
        s = weight_term * _sim(q, r.get("term","")) + weight_notes * _sim(q, r.get("notes",""))
        scored.append((s, r))
    scored.sort(key=lambda x: x[0], reverse=True)
    out = [{"score": round(s, 4), "row": r} for s, r in scored[:max(1,int(top_k))]]
    return {
        "status": "ok" if out else "empty",
        "files_scanned": res.get("files_scanned", []),
        "query": q,
        "top_k": top_k,
        "results": out
    }

@mcp.tool()
def sense_import_md(
    path: str,
    file: Optional[str] = None,           # force all rows into a single CSV
    on_duplicate: str = "skip",           # skip|update|error
    dry_run: bool = True
) -> Dict:
    """
    Read a Markdown table and import rows. The MD must have a header row with at least the columns in HEADER.
    Escaped '|' in cells are supported (\\|).
    """
    p = Path(path)
    if not p.is_absolute():
        # allow relative to exports/ or project root
        p = (EXPORT_DIR / p) if (EXPORT_DIR / p).exists() else (ROOT / p)
    if not p.exists():
        return {"status":"error","error":"file_not_found","path":str(p)}

    # read lines and find header/separator
    lines = p.read_text(encoding="utf-8").splitlines()
    header_idx = -1
    cols: List[str] = []
    for i, ln in enumerate(lines):
        if ln.strip().startswith("|") and ("|" in ln):
            cells = [c.strip().lower() for c in _md_split_row(ln)]
            if set(HEADER).issubset(set(cells)):
                header_idx = i
                cols = cells
                break
    if header_idx < 0:
        return {"status":"error","error":"header_not_found","required":HEADER}

    # parse body until a blank or EOF
    records: List[Dict] = []
    for ln in lines[header_idx+1:]:
        if not ln.strip(): break
        # skip separator like |---|---|
        if set(ln.replace("|","").replace("-","").strip()) == set():
            continue
        cells = _md_split_row(ln)
        if len(cells) != len(cols):
            # tolerate shorter/longer rows by padding/truncating
            if len(cells) < len(cols):
                cells += [""] * (len(cols) - len(cells))
            else:
                cells = cells[:len(cols)]
        row = {cols[i]: cells[i] for i in range(len(cols))}
        # normalize to our header keys
        rec = {
            "term": row.get("term",""),
            "category": (row.get("category","") or "").lower(),
            "locale": row.get("locale",""),
            "era": row.get("era",""),
            "weather": (row.get("weather","") or "any").lower(),
            "register": (row.get("register","") or "common").lower(),
            "notes": row.get("notes",""),
        }
        # sanitize
        for k in rec: rec[k] = _sanitize(rec[k])
        records.append(rec)

    # route per-record to destination file(s)
    added = 0
    updated = 0
    skipped = 0
    errors: List[Dict] = []
    files_touched: set[str] = set()

    # cache rows per target file so we can update/append once
    file_rows: Dict[Path, List[Dict]] = {}

    def target_path(locale_val: str) -> Path:
        chosen = Path(file).name if file else _default_file_for_locale(locale_val)
        return DATA_DIR / chosen

    # preload for update modes
    targets = {target_path(r["locale"]) for r in records}
    for tp in targets:
        file_rows[tp] = _load_rows(tp)

    for rec in records:
        tp = target_path(rec["locale"])
        rows = file_rows[tp]
        key = _row_key(rec)
        idx = next((i for i,r in enumerate(rows) if _row_key(r)==key), -1)
        if idx >= 0:
            if on_duplicate == "skip":
                skipped += 1; continue
            elif on_duplicate == "error":
                errors.append({"file":str(tp),"issue":"duplicate","row":rec}); continue
            elif on_duplicate == "update":
                rows[idx] = rec; updated += 1
        else:
            rows.append(rec); added += 1
        files_touched.add(str(tp))

    if not dry_run:
        for tp, rows in file_rows.items():
            _write_rows(tp, rows)

    return {
        "status": "dry_run" if dry_run else "imported",
        "path": str(p),
        "added": added,
        "updated": updated,
        "skipped": skipped,
        "errors": errors,
        "files_touched": sorted(files_touched)
    }

@mcp.tool()
def sense_suggest(
    locale: str, era: str, weather: str = "any",
    register: str = "common", n: int = 6,
    memory_key: Optional[str] = None,
    exclude: Optional[str] = None,          # e.g. "smell,sound"
    exclude_terms: Optional[str] = None,    # e.g. "camellia hair oil, inkstone and soot"
    no_repeat: bool = False,                # exclude memory.used_terms if memory_key set
    update_memory: bool = True              # internal: False when called from rewrite
) -> List[Dict]:
    rows = load_corpus(str(DATA_DIR))
    memory = load_memory(str(MEM_DIR), memory_key) if memory_key else {}

    excl_cats  = _split_set(exclude)
    excl_terms = _split_set(exclude_terms)
    if no_repeat and memory_key:
        excl_terms |= {t.strip().lower() for t in memory.get("used_terms", [])}

    # Ask for more than needed, then filter down.
    raw = suggest_sensory(
        rows=rows, locale=locale, era=era,
        weather=weather, register=register,
        n=max(n * 5, n), memory=memory
    )

    picked: List[Dict] = []
    seen_terms: set[str] = set()
    for x in raw:
        cat = (x.get("category","") or "").strip().lower()
        term = (x.get("term","") or "").strip()
        if cat in excl_cats:        continue
        if term.lower() in excl_terms: continue
        if term.lower() in seen_terms: continue  # belt-and-suspenders
        picked.append(x)
        seen_terms.add(term.lower())
        if len(picked) >= n: break

    items = picked if picked else raw[:n]

    if update_memory and memory_key:
        used = memory.get("used_terms", [])
        used.extend([x["term"] for x in items])
        memory["used_terms"] = sorted(set(used))
        save_memory(str(MEM_DIR), memory_key, memory)

    return items


@mcp.tool()
def sense_rewrite(
    locale: str, era: str, text: str, weather: str = "any",
    register: str = "common", n: int = 6,
    memory_key: Optional[str] = None,
    exclude: Optional[str] = None,          # new
    exclude_terms: Optional[str] = None,    # new
    no_repeat: bool = False                 # new
) -> Dict:
    # Pull suggestions with the same filters, but don't double-write memory here.
    suggestions = sense_suggest(
        locale=locale, era=era, weather=weather, register=register, n=n,
        memory_key=memory_key, exclude=exclude, exclude_terms=exclude_terms,
        no_repeat=no_repeat, update_memory=False
    )

    out = rewrite_with_sensory(text, suggestions)
    result = {
        "suggestions": suggestions,
        "rewrite": out.get("rewrite"),
        "model": out.get("model"),
        "reasoning": out.get("reasoning")
    }

    # Now write memory once (for no_repeat across turns)
    if memory_key:
        memory = load_memory(str(MEM_DIR), memory_key) or {}
        used = memory.get("used_terms", [])
        used.extend([x["term"] for x in suggestions])
        memory["used_terms"] = sorted(set(used))
        save_memory(str(MEM_DIR), memory_key, memory)

    return result


@mcp.tool()
def sense_edit(
        match_term: str, match_category: str, match_locale: str, match_era: str,
        term: Optional[str] = None, category: Optional[str] = None,
        locale: Optional[str] = None, era: Optional[str] = None,
        weather: Optional[str] = None, register: Optional[str] = None,
        notes: Optional[str] = None, file: Optional[str] = None,
        dry_run: bool = False
) -> Dict:
    """Edit a row identified by (match_term, match_category, match_locale, match_era)."""
    target_key = _row_key({
        "term": match_term, "category": match_category,
        "locale": match_locale, "era": match_era
    })
    path = DATA_DIR / (Path(file).name if file else _default_file_for_locale(match_locale))
    if not path.exists():
        found = _find_in_any_csv(target_key) if not file else None
        if found is None:
            return {"status": "not_found", "file": str(path)}
        path = found

    rows = _load_rows(path)
    idx = next((i for i, r in enumerate(rows) if _row_key(r) == target_key), -1)
    if idx < 0:
        return {"status": "not_found", "file": str(path)}

    before = dict(rows[idx])
    after = dict(before)

    def apply_if(val, key, normalize_lower=False):
        if val is None: return
        v = _sanitize(val)
        after[key] = v.lower() if normalize_lower else v

    apply_if(term, "term")
    apply_if(category, "category", True)
    apply_if(locale, "locale")
    apply_if(era, "era")
    apply_if(weather, "weather", True)
    apply_if(register, "register", True)
    apply_if(notes, "notes")

    # if identity changed, check duplicate
    if _row_key(before) != _row_key(after):
        if any(_row_key(r) == _row_key(after) for r in rows):
            return {"status": "duplicate", "file": str(path), "before": before, "after": after}

    if dry_run:
        return {"status": "dry_run", "file": str(path), "before": before, "after": after}

    rows[idx] = after
    _write_rows(path, rows)
    return {"status": "edited", "file": str(path), "before": before, "after": after}


@mcp.tool()
def sense_delete(
        match_term: str, match_category: str, match_locale: str, match_era: str,
        file: Optional[str] = None, dry_run: bool = False
) -> Dict:
    """Delete rows matching the composite key. If file is omitted, searches all CSVs."""
    target_key = _row_key({
        "term": match_term, "category": match_category,
        "locale": match_locale, "era": match_era
    })

    candidates = [DATA_DIR / Path(file).name] if file else list(DATA_DIR.glob("*.csv"))
    total_removed = 0
    removed_from: List[str] = []

    for path in candidates:
        rows = _load_rows(path)
        keep = [r for r in rows if _row_key(r) != target_key]
        removed = len(rows) - len(keep)
        if removed > 0:
            removed_from.append(str(path))
            total_removed += removed
            if not dry_run:
                _write_rows(path, keep)

    if total_removed == 0:
        return {"status": "not_found", "removed": 0}
    return {"status": "deleted" if not dry_run else "dry_run",
            "removed": total_removed, "files": removed_from}


# NEW: list/filter rows across your CSVs
@mcp.tool()
def sense_list(
        category: Optional[str] = None,
        locale: Optional[str] = None,
        era: Optional[str] = None,
        weather: Optional[str] = None,
        register: Optional[str] = None,
        term_contains: Optional[str] = None,
        notes_contains: Optional[str] = None,
        file: Optional[str] = None,
        limit: int = 50,
        sort: str = "category,term"
) -> Dict:
    """
    Return rows matching simple filters. All matches are case-insensitive.
    - Equality filters: category, locale, era, weather, register
    - Substring filters: term_contains, notes_contains
    - file: restrict to a single CSV; otherwise scan all CSVs
    - sort: comma-separated fields (subset of HEADER), e.g. "category,term"
    """

    def _match(row: Dict) -> bool:
        def eq(val, field):
            return (val is None) or (row[field].strip().lower() == val.strip().lower())

        def contains(val, field):
            return (val is None) or (val.strip().lower() in row[field].strip().lower())

        return (eq(category, "category") and
                eq(locale, "locale") and
                eq(era, "era") and
                eq(weather, "weather") and
                eq(register, "register") and
                contains(term_contains, "term") and
                contains(notes_contains, "notes"))

    paths = [DATA_DIR / Path(file).name] if file else list(DATA_DIR.glob("*.csv"))
    matches: List[Dict] = []
    for p in paths:
        for r in _load_rows(p):
            if _match(r):
                matches.append(r)

    # sort
    fields = [f.strip() for f in sort.split(",") if f.strip() in HEADER]
    if fields:
        matches.sort(key=lambda d: tuple(d[f].strip().lower() for f in fields))

    out = {
        "count": len(matches),
        "files_scanned": [str(p) for p in paths],
        "items": matches[:max(0, int(limit))] if limit else matches
    }
    return out


@mcp.tool()
def sense_add(
    term: str,
    category: str,
    locale: str,
    era: str,
    weather: str = "any",
    register: str = "common",
    notes: str = "",
    file: Optional[str] = None,
) -> Dict:
    """
    Append a new sensory row to data/*.csv.

    - Dedupes on (term, category, locale, era).
    - Creates the CSV if missing and writes the header.
    - If `file` not given, routes to a default file per locale (e.g., jp_sensory.csv).

    Returns (as MCP JSON content via _json):
      {"status": "added"|"exists"|"error", "file": "...", "record": {...}, "created_file": bool}
    """
    try:
        record = {
            "term": _sanitize(term),
            "category": _sanitize(category).lower(),
            "locale": _sanitize(locale),
            "era": _sanitize(era),
            "weather": (_sanitize(weather) or "any").lower(),
            "register": (_sanitize(register) or "common").lower(),
            "notes": _sanitize(notes),
        }

        # choose a target CSV
        chosen = Path(file).name if file else _default_file_for_locale(record["locale"])
        path = DATA_DIR / chosen
        path.parent.mkdir(parents=True, exist_ok=True)

        # dedupe
        existing = _load_rows(path)
        if any(_row_key(r) == _row_key(record) for r in existing):
            return _json({
                "status": "exists",
                "file": str(path),
                "record": record,
                "created_file": False,
            })

        # append (header if new/empty)
        created_file = (not path.exists()) or (path.stat().st_size == 0)
        with path.open("a", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(cast(SupportsWriteStr, f), fieldnames=HEADER)
            if created_file:
                w.writeheader()
            w.writerow(record)

        result = {"status": "added", "file": str(path), "record": record, "created_file": created_file}
        return _json(result)

    except Exception as e:
        # never return an empty result to the client
        return _json({"status": "error", "error": str(e)})

@mcp.tool()
def sense_export(
    category: Optional[str] = None,
    locale: Optional[str] = None,
    era: Optional[str] = None,
    weather: Optional[str] = None,
    register: Optional[str] = None,
    term_contains: Optional[str] = None,
    notes_contains: Optional[str] = None,
    file: Optional[str] = None,    # restrict search to a single CSV; else scan all
    limit: int = 0,                 # 0 = all matches
    sort: str = "category,term",
    out: Optional[str] = None,      # output path; if relative -> exports/
    append: bool = False,
    include_header: bool = True
) -> Dict:
    """
    Export matching rows to a CSV file (uses sense_list filters).
    """
    import csv
    from datetime import datetime
    from pathlib import Path
    from typing import Literal, Protocol, cast

    class _Writer(Protocol):
        def write(self, s: str) -> object: ...

    # 1) Collect matches via sense_list (keeps one source of truth)
    results = sense_list(
        category=category, locale=locale, era=era,
        weather=weather, register=register,
        term_contains=term_contains, notes_contains=notes_contains,
        file=file, limit=limit, sort=sort
    )
    rows = results.get("items", [])
    filters = {
        "category": category, "locale": locale, "era": era,
        "weather": weather, "register": register,
        "term_contains": term_contains, "notes_contains": notes_contains,
        "file": file, "limit": limit, "sort": sort
    }
    if not rows:
        return {"status": "empty", "written": 0, "file": None,
                "filters": filters, "files_scanned": results.get("files_scanned", [])}

    # 2) Resolve output path
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    if out:
        path = Path(out)
        if not path.is_absolute():
            path = EXPORT_DIR / path
    else:
        def clean(x: Optional[str]) -> str:
            return (x or "all").replace("/", "_").replace(" ", "_")
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = EXPORT_DIR / f"sense_export_{clean(locale)}_{clean(era)}_{clean(category)}_{stamp}.csv"

    if path.suffix.lower() != ".csv":
        path = path.with_suffix(".csv")

    # 3) Write/append CSV (type-checker friendly)
    mode: Literal["a", "w"] = "a" if append else "w"
    write_header = (
            include_header and (
            mode == "w" or
            not path.exists() or
            (path.exists() and path.stat().st_size == 0)
    )
    )
    with path.open(mode, encoding="utf-8", newline="") as f:
        w = csv.DictWriter(cast(SupportsWriteStr, f), fieldnames=HEADER)
        if write_header:
            w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in HEADER})

    return {
        "status": "exported",
        "written": len(rows),
        "file": str(path),
        "append": append,
        "filters": filters,
        "files_scanned": results.get("files_scanned", [])
    }


@mcp.tool()
def sense_export_sheet(
    category: Optional[str] = None,
    locale: Optional[str] = None,
    era: Optional[str] = None,
    weather: Optional[str] = None,
    register: Optional[str] = None,
    term_contains: Optional[str] = None,
    notes_contains: Optional[str] = None,
    file: Optional[str] = None,
    limit: int = 0,                 # 0 = all matches
    sort: str = "category,term",
    out: Optional[str] = None,      # relative path -> exports/
    sheet_name: str = "SenseBank",
    include_header: bool = True,
    overwrite: bool = False,
) -> Dict:
    """
    Export matching rows to an XLSX workbook (one sheet). Requires: pip install xlsxwriter.
    If `out` exists and overwrite=False, a timestamp is appended to the filename.
    """
    from datetime import datetime
    from pathlib import Path
    from typing import Any, Optional

    result: Dict[str, object] = {}
    files_scanned: list[str] = []
    error: Optional[str] = None

    # 0) dependency check (bind alias so linters know it's defined)
    xlsxwriter_mod: Any = None
    try:
        import xlsxwriter as _xlsxwriter  # type: ignore
        xlsxwriter_mod = _xlsxwriter
    except Exception as e:
        error = f"xlsxwriter not installed. pip install xlsxwriter ({e})"

    # 1) collect rows
    rows: list[Dict] = []
    if error is None:
        r = sense_list(
            category=category, locale=locale, era=era,
            weather=weather, register=register,
            term_contains=term_contains, notes_contains=notes_contains,
            file=file, limit=limit, sort=sort
        )
        rows = r.get("items") or []
        files_scanned = r.get("files_scanned", [])

        if not rows:
            result = {"status": "empty", "written": 0, "file": None, "files_scanned": files_scanned}

    # 2) resolve output path
    if error is None and not result:
        EXPORT_DIR.mkdir(parents=True, exist_ok=True)

        def _clean(x: Optional[str]) -> str:
            return (x or "all").replace("/", "_").replace(" ", "_")

        path = Path(out) if out else EXPORT_DIR / (
            f"sense_export_{_clean(locale)}_{_clean(era)}_{_clean(category)}_{datetime.now():%Y%m%d_%H%M%S}.xlsx"
        )
        if not path.is_absolute():
            path = EXPORT_DIR / path
        if path.suffix.lower() != ".xlsx":
            path = path.with_suffix(".xlsx")
        if path.exists() and not overwrite:
            path = path.with_name(path.stem + f"_{datetime.now():%Y%m%d_%H%M%S}" + path.suffix)

        # 3) write workbook (context manager + proven binding)
        sheet = (sheet_name or "SenseBank")[:31]
        try:
            assert xlsxwriter_mod is not None  # reassure the analyzer
            with xlsxwriter_mod.Workbook(str(path)) as wb:
                ws = wb.add_worksheet(sheet)
                row_i = 0
                if include_header:
                    for col_i, key in enumerate(HEADER):
                        ws.write(row_i, col_i, key)
                    row_i += 1
                for row in rows:
                    for col_i, key in enumerate(HEADER):
                        ws.write(row_i, col_i, row.get(key, ""))
                    row_i += 1
            result = {
                "status": "exported",
                "written": len(rows),
                "file": str(path),
                "sheet": sheet,
                "files_scanned": files_scanned,
            }
        except Exception as e:
            # best-effort cleanup of empty partial
            try:
                if path.exists() and path.stat().st_size == 0:
                    path.unlink()
            except Exception:
                pass
            error = str(e)

    # 4) single tail return
    if error is not None and not result:
        result = {"status": "error", "error": error, "files_scanned": files_scanned}

    # If you use a JSON wrapper for MCP, apply it here:
    try:
        return _json(result)  # type: ignore[name-defined]
    except Exception:
        return result



@mcp.tool()
def sense_export_md(
    category: Optional[str] = None,
    locale: Optional[str] = None,
    era: Optional[str] = None,
    weather: Optional[str] = None,
    register: Optional[str] = None,
    term_contains: Optional[str] = None,
    notes_contains: Optional[str] = None,
    file: Optional[str] = None,
    limit: int = 0,                 # 0 = all matches
    sort: str = "category,term",
    out: Optional[str] = None,      # relative path -> exports/
    append: bool = False,
    include_header: bool = True
) -> Dict:
    """
    Export matching rows to a Markdown table (.md).
    Returns: {status, written, file, append, header_written, files_scanned}
    """
    from typing import Literal, cast
    from datetime import datetime

    # Gather matches via sense_list (single source of truth for filters/sort)
    results = sense_list(
        category=category, locale=locale, era=era,
        weather=weather, register=register,
        term_contains=term_contains, notes_contains=notes_contains,
        file=file, limit=limit, sort=sort
    )
    rows = results.get("items", [])
    if not rows:
        return {"status": "empty", "written": 0, "file": None, "files_scanned": results.get("files_scanned", [])}

    # Resolve output path
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    if out:
        path = Path(out)
        if not path.is_absolute():
            path = EXPORT_DIR / path
    else:
        def clean(x: Optional[str]) -> str:
            return (x or "all").replace("/", "_").replace(" ", "_")
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = EXPORT_DIR / f"sense_export_{clean(locale)}_{clean(era)}_{clean(category)}_{stamp}.md"

    if path.suffix.lower() != ".md":
        path = path.with_suffix(".md")

    # Markdown header + escape helper
    header_line = "| " + " | ".join(HEADER) + " |\n"
    sep_line    = "| " + " | ".join("---" for _ in HEADER) + " |\n"

    def md_escape(s: str) -> str:
        # escape pipe; newlines are already sanitized on ingest
        return (s or "").replace("|", "\\|")

    # Write/append
    mode: Literal["a","w"] = "a" if append else "w"
    write_header = include_header or (mode == "w") or (not path.exists()) or (path.stat().st_size == 0)
    with path.open(mode, encoding="utf-8") as f:
        wf = cast(SupportsWriteStr, f)
        if write_header:
            wf.write(header_line)
            wf.write(sep_line)
        for r in rows:
            wf.write("| " + " | ".join(md_escape(r.get(k, "")) for k in HEADER) + " |\n")

    return {
        "status": "exported",
        "written": len(rows),
        "file": str(path),
        "append": append,
        "header_written": write_header,
        "files_scanned": results.get("files_scanned", [])
    }

@mcp.tool()
def sense_validate(
    file: Optional[str] = None,
    fix_whitespace: bool = False,     # trim spaces, collapse newlines
    normalize_case: bool = False,     # lowercase category/weather/register
    dedupe: bool = False,             # remove exact dupes within each file
    global_dedupe: bool = False,      # report dupes across files (report-only)
    allowed_categories: Optional[str] = None,  # comma-separated; optional
    allowed_registers: Optional[str] = None,   # comma-separated; optional
    allowed_weather: Optional[str] = None,     # comma-separated; optional
    dry_run: bool = True,             # when False, applies fixes/dedupes
    limit: int = 100                  # cap number of problems returned
) -> Dict:
    """
    Validate rows. Returns a summary and up to `limit` problems.
    Deduping is per-file only; cross-file dupes are reported (not removed).
    """
    def norm(s: str) -> str: return (s or "").strip()
    def lower(s: str) -> str: return norm(s).lower()

    paths = [DATA_DIR / Path(file).name] if file else list(DATA_DIR.glob("*.csv"))
    allowed_cat = set(map(lower, (allowed_categories or "").split(","))) - {""}
    allowed_reg = set(map(lower, (allowed_registers or "").split(","))) - {""}
    allowed_wth = set(map(lower, (allowed_weather or "").split(","))) - {""}

    # If not provided, fall back to defaults (comment out if you prefer opt-in)
    if not allowed_reg: allowed_reg = set(DEFAULT_ALLOWED_REGISTERS)
    if not allowed_wth: allowed_wth = set(DEFAULT_ALLOWED_WEATHER)

    problems: List[Dict] = []
    files_scanned: List[str] = []
    rows_scanned = 0
    files_modified = 0

    # track cross-file duplicates
    global_seen: Dict[Tuple[str,str,str,str], str] = {}

    for path in paths:
        rows = _load_rows(path)
        files_scanned.append(str(path))
        rows_scanned += len(rows)

        # in-file dedupe tracker
        seen_keys: set = set()
        new_rows: List[Dict] = []
        modified = False

        for idx, r in enumerate(rows):
            original = dict(r)

            # whitespace normalization
            if fix_whitespace:
                for k in HEADER:
                    r[k] = _sanitize(r.get(k, ""))

            # case normalization for enum-ish fields
            if normalize_case:
                r["category"] = lower(r.get("category", ""))
                r["weather"]  = lower(r.get("weather", ""))
                r["register"] = lower(r.get("register", ""))

            # check required core fields
            missing = [k for k in ("term","category","locale","era") if not norm(r.get(k,""))]
            if missing:
                problems.append({"file": str(path), "index": idx, "issue": "missing_required", "fields": missing, "row": r})

            # check allowed sets (if provided)
            if allowed_cat and r.get("category"):
                if lower(r["category"]) not in allowed_cat:
                    problems.append({"file": str(path), "index": idx, "issue": "bad_category", "value": r["category"], "allowed": sorted(allowed_cat)})
            if allowed_reg and r.get("register"):
                if lower(r["register"]) not in allowed_reg:
                    problems.append({"file": str(path), "index": idx, "issue": "bad_register", "value": r["register"], "allowed": sorted(allowed_reg)})
            if allowed_wth and r.get("weather"):
                if lower(r["weather"]) not in allowed_wth:
                    problems.append({"file": str(path), "index": idx, "issue": "bad_weather", "value": r["weather"], "allowed": sorted(allowed_wth)})

            key = _row_key(r)

            # cross-file duplicate discovery (report-only)
            if global_dedupe:
                first = global_seen.get(key)
                if first and first != str(path):
                    problems.append({"issue": "duplicate_across_files", "first_file": first, "dup_file": str(path), "row": r})
                else:
                    global_seen[key] = str(path)

            # in-file dedupe (remove subsequent duplicates)
            if dedupe:
                if key in seen_keys:
                    modified = True
                    # drop duplicate
                else:
                    seen_keys.add(key)
                    new_rows.append(r)
            else:
                new_rows.append(r)

            if r != original:
                modified = True

        if modified and not dry_run:
            _write_rows(path, new_rows)
            files_modified += 1

    return {
        "status": ("issues" if problems else "ok") if dry_run else ("fixed" if files_modified else ("issues" if problems else "ok")),
        "files_scanned": files_scanned,
        "rows_scanned": rows_scanned,
        "files_modified": files_modified if not dry_run else 0,
        "problems_count": len(problems),
        "problems": problems[:max(0,int(limit))],
        "notes": "Cross-file duplicates are reported only (not removed). Set dedupe=True to remove in-file duplicates."
    }

@mcp.tool()
def sense_move(
    match_term: str,
    match_category: str,
    match_locale: str,
    match_era: str,
    dest_locale: Optional[str] = None,         # choose target file by locale; also update row.locale if update_locale_field=True
    dest_file: Optional[str] = None,           # explicit target file overrides dest_locale
    update_locale_field: bool = True,          # change the 'locale' value in the row when dest_locale is provided
    dry_run: bool = True
) -> Dict:
    """
    Find row by (term, category, locale, era), remove it from its source CSV,
    and append to destination CSV. Prevents duplicates at dest.
    """
    key = _row_key({"term": match_term, "category": match_category, "locale": match_locale, "era": match_era})

    # locate source file
    src_path = _find_in_any_csv(key)
    if src_path is None:
        return {"status": "not_found", "reason": "source_row_not_found"}

    # choose destination
    if dest_file:
        dst_path = DATA_DIR / Path(dest_file).name
    elif dest_locale:
        dst_path = DATA_DIR / _default_file_for_locale(dest_locale)
    else:
        # if nothing given, default to file for existing locale (no-op move unless you really want to consolidate)
        dst_path = DATA_DIR / _default_file_for_locale(match_locale)

    # load rows
    src_rows = _load_rows(src_path)
    dst_rows = _load_rows(dst_path)

    # find row in source
    idx = next((i for i, r in enumerate(src_rows) if _row_key(r) == key), -1)
    if idx < 0:
        return {"status": "not_found", "reason": "source_row_not_found_again"}

    row = dict(src_rows[idx])
    before = dict(row)

    # update locale field if requested
    if dest_locale and update_locale_field:
        row["locale"] = dest_locale

    # prevent duplicate at destination
    if any(_row_key(r) == _row_key(row) for r in dst_rows):
        return {"status": "duplicate", "from_file": str(src_path), "to_file": str(dst_path), "row": row}

    result = {
        "status": "dry_run" if dry_run else "moved",
        "from_file": str(src_path),
        "to_file": str(dst_path),
        "before": before,
        "after": row,
    }

    if dry_run:
        return result

    # commit: remove from src, append to dest
    del src_rows[idx]
    dst_rows.append(row)
    _write_rows(src_path, src_rows)
    _write_rows(dst_path, dst_rows)
    return result


@mcp.tool()
def sense_stats(
    # Same filters as sense_list
    category: Optional[str] = None,
    locale: Optional[str] = None,
    era: Optional[str] = None,
    weather: Optional[str] = None,
    register: Optional[str] = None,
    term_contains: Optional[str] = None,
    notes_contains: Optional[str] = None,
    file: Optional[str] = None,
    # Grouping & output
    group_by: str = "locale,era,category",   # CSV of HEADER fields
    sort_by: str = "count",                  # "count" | "unique_terms" | "group"
    desc: bool = True,                       # sort descending
    top: int = 0,                            # 0 = all groups
    examples: bool = True,                   # include example terms per group
    examples_per: int = 3                    # how many examples per group
) -> Dict:
    """
    Summarize the corpus with flexible grouping.
    Returns:
      {
        status, files_scanned, total_rows, group_by,
        groups: [
          { key:{...}, count:int, unique_terms:int, examples?: [str, ...] },
          ...
        ]
      }
    """
    # 1) Pull rows using the single source of truth
    results = sense_list(
        category=category, locale=locale, era=era,
        weather=weather, register=register,
        term_contains=term_contains, notes_contains=notes_contains,
        file=file, limit=0, sort="category,term",  # limit=0 => no cap
    )
    items: List[Dict[str, Any]] = results.get("items", [])
    files_scanned: List[str] = results.get("files_scanned", [])
    total_rows = len(items)

    # 2) Validate group_by fields
    fields: List[str] = [f.strip() for f in group_by.split(",") if f.strip()]
    fields = [f for f in fields if f in HEADER]
    if not fields:
        fields = ["category"]  # sensible fallback

    # 3) Aggregate
    Key = Tuple[str, ...]
    counts: Dict[Key, int] = defaultdict(int)
    uniq_terms: Dict[Key, set] = defaultdict(set)
    samples: Dict[Key, List[str]] = defaultdict(list)
    seen_in_sample: Dict[Key, set] = defaultdict(set)

    examples_n = max(0, int(examples_per))

    for r in items:
        key_tuple: Key = tuple(str(r.get(f, "") or "") for f in fields)
        counts[key_tuple] += 1
        term = str(r.get("term", "") or "")
        if term:
            uniq_terms[key_tuple].add(term)
            if examples and term not in seen_in_sample[key_tuple]:
                if len(samples[key_tuple]) < examples_n:
                    samples[key_tuple].append(term)
                    seen_in_sample[key_tuple].add(term)

    # 4) Build group rows
    groups: List[Dict[str, Any]] = []
    for key_tuple, cnt in counts.items():
        row: Dict[str, Any] = {
            "key": {fname: key_tuple[i] for i, fname in enumerate(fields)},
            "count": int(cnt),
            "unique_terms": int(len(uniq_terms[key_tuple])),
        }
        if examples:
            row["examples"] = list(samples.get(key_tuple, []))
        groups.append(row)

    # 5) Sort
    sb = (sort_by or "count").lower()
    if sb == "count":
        groups.sort(key=lambda g: g["count"], reverse=bool(desc))
    elif sb in ("unique", "unique_terms"):
        groups.sort(key=lambda g: g["unique_terms"], reverse=bool(desc))
    else:
        # lexicographic by group key values (case-insensitive)
        groups.sort(
            key=lambda g: tuple((g["key"].get(f, "") or "").lower() for f in fields),
            reverse=bool(desc),
        )

    # 6) Truncate
    if int(top) > 0:
        groups = groups[:int(top)]

    return {
        "status": "ok" if groups else "empty",
        "files_scanned": files_scanned,
        "total_rows": total_rows,
        "group_by": fields,
        "filters": {
            "category": category, "locale": locale, "era": era,
            "weather": weather, "register": register,
            "term_contains": term_contains, "notes_contains": notes_contains,
            "file": file
        },
        "sort_by": sb,
        "desc": bool(desc),
        "groups": groups,
    }


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
