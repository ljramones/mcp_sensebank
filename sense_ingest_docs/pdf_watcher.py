#!/usr/bin/env python3
"""
PDF processing and directory watching functionality.
Handles PDF text extraction, file monitoring, and batch processing.
"""

import time
import logging
import shutil
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Counter
from contextlib import contextmanager

from sense_ingest_docs.context_inference import load_rules, infer_context
from sense_ingest_docs.csv_data_manager import _csv_add_record_direct
from sense_ingest_docs.llm_interface import extract_sensory_terms
from sense_ingest_docs.mcp_client import sense_add
from sense_ingest_docs.reporting import summarize_cats

DEFAULT_REGISTER = "common"
DEFAULT_WEATHER = "any"

MIN_TERM_CHARS = 6  # Increased for better quality


ALLOWED_CATS = {"smell", "sound", "taste", "touch", "sight"}


# Updated ban terms - comprehensive filtering
BAN_TERMS = {
    # Body parts
    "face", "eyes", "eye", "hand", "hands", "arm", "arms", "body", "head", "neck", "chest", "mouth", "lips", "nose",
    # Senses themselves
    "sight", "sound", "taste", "touch", "smell", "hearing", "vision",
    # Emotions (not sensory)
    "fear", "anger", "joy", "sadness", "disgust", "surprise", "calm", "panic", "hatred", "love",
    # Generic objects
    "voice", "expression", "package", "window", "doll", "jar", "bucket", "water",
    # Lighting terms (too generic)
    "light", "darkness", "bright", "dark",
    # Movement/actions
    "motion", "movement", "grabbing", "running", "walking",
    # Generic descriptors
    "people", "person", "man", "woman", "child", "children"
}



def extract_items_standard(chunk: str, ctx: Dict) -> List[Dict]:
    """Phase 1 extraction with basic validation."""
    logger = logging.getLogger("sense-ingest")

    # Extract using LLM interface
    from sense_ingest_docs.llm_interface import extract_sensory_terms

    raw_items = extract_sensory_terms(
        chunk=chunk,
        locale=ctx.get("locale", ""),
        era=ctx.get("era", ""),
        register=ctx.get("register", "common"),
        weather=ctx.get("weather", "any")
    )

    # Apply validation logic
    requirements = get_category_requirements()
    validated = []
    rejection_reasons = []

    for item in raw_items:
        term = (item.get("term", "") or "").strip()
        category = (item.get("category", "") or "").strip().lower()
        notes = (item.get("notes", "") or "").strip()

        # Basic validation
        if not term or category not in ALLOWED_CATS:
            continue

        term_lower = term.lower()

        # Apply Phase 1 filters
        if any(banned in term_lower for banned in BAN_TERMS):
            rejection_reasons.append(f"'{term}': banned term")
            continue

        # Emotion/abstract check
        banned_indicators = [
            "annoyed", "angry", "furious", "confident", "nervous", "sad", "happy",
            "glance", "look", "stare", "air", "expression", "manner"
        ]
        if any(banned in term_lower for banned in banned_indicators):
            rejection_reasons.append(f"'{term}': emotion/abstract concept")
            continue

        # Physiological check
        physio_terms = ["breath", "heartbeat", "pulse", "circulation"]
        if any(physio in term_lower for physio in physio_terms):
            rejection_reasons.append(f"'{term}': physiological rather than sensory")
            continue

        # Category-specific validation
        cat_req = requirements[category]
        has_required = (
                any(word in term_lower for word in cat_req["required_words"]) or
                any(concept in term_lower for concept in cat_req["required_concepts"])
        )
        if not has_required:
            rejection_reasons.append(f"'{term}': {cat_req['description']}")
            continue

        # Length check
        if len(term) < MIN_TERM_CHARS:
            rejection_reasons.append(f"'{term}': too short")
            continue

        # All validation passed
        validated.append({
            "term": term.title(),
            "category": category,
            "notes": notes
        })

    # Logging
    logger.info(f"Standard extraction validated {len(validated)}/{len(raw_items)} terms")
    if validated:
        logger.info(f"Valid terms: {', '.join(item['term'] for item in validated[:3])}")
    if rejection_reasons and logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"Standard filtering: {'; '.join(rejection_reasons[:3])}")

    return validated



# ------------------ PDF text extraction ------------------
def pdf_to_texts(pdf_path: Path, ocr: bool = True, ocr_lang: str = "eng") -> List[str]:
    """
    Converts a PDF file into a list of text strings, where each string corresponds
    to the extracted text of a PDF page. Optionally supports OCR for pages with
    low text extraction reliability.

    :param pdf_path: The file path to the PDF document. This is used as an input
        to extract text from each PDF page.
    :type pdf_path: Path
    :param ocr: Boolean flag indicating whether to perform Optical Character
        Recognition (OCR) on pages where text extraction results are insufficient.
        Defaults to True.
    :type ocr: bool
    :param ocr_lang: Language setting for OCR, following the codes used by
        Tesseract OCR (e.g., "eng" for English). Defaults to "eng".
    :type ocr_lang: str
    :return: A list of strings, where each string contains the extracted or OCR-processed
        text from a corresponding page in the PDF.
    :rtype: List[str]
    """
    logger = logging.getLogger("sense-ingest")
    texts: List[str] = []
    try:
        import pdfplumber
    except Exception as e:
        logger.error("pdfplumber missing: %s", e)
        return []

    with log_timing(logger, f"open {pdf_path.name}"):
        with pdfplumber.open(str(pdf_path)) as pdf:
            n_pages = len(pdf.pages)
            logger.info("Pages: %d", n_pages)
            for i, page in enumerate(pdf.pages, 1):
                t = (page.extract_text() or "").strip()
                if t:
                    logger.debug("Page %d extracted %d chars", i, len(t))
                if ocr and len(t) < 40:
                    try:
                        from pdf2image import convert_from_path
                        import pytesseract
                        logger.debug("Page %d low text → OCR...", i)
                        img = convert_from_path(str(pdf_path), first_page=i, last_page=i)[0]
                        ot = pytesseract.image_to_string(img, lang=ocr_lang) or ""
                        t = (ot.strip() or t)
                        logger.debug("Page %d OCR result %d chars", i, len(t))
                    except Exception as e:
                        logger.warning("OCR failed on page %d: %s", i, e)
                texts.append(t)
    return texts


# ------------------ File utilities ------------------
def _is_file_stable(p: Path, wait: float = 0.5, checks: int = 4) -> bool:
    """
    Treat a file as 'ready' only if its size+mtime stop changing.
    This avoids ingesting while a copy is still in progress.
    """
    try:
        last = None
        for _ in range(checks):
            st = p.stat()
            cur = (st.st_size, st.st_mtime_ns)
            if cur == last:
                return True
            last = cur
            time.sleep(wait)
    except FileNotFoundError:
        return False
    return False


def _list_pdfs(d: Path) -> list[Path]:
    """
    Return non-hidden PDFs in d (case-insensitive on extension), skipping temporary/marker files.
    """
    # Use a case-insensitive glob
    pdfs = set()
    for pat in ("*.pdf", "*.PDF", "*.[Pp][Dd][Ff]"):
        pdfs.update(d.glob(pat))

    out = []
    for p in pdfs:
        name = p.name
        if name.startswith("."):  # hidden
            continue
        if name.endswith(".processing"):  # our in-progress marker (if you add it)
            continue
        # sometimes weird extensions like '.pdf ' exist; normalize by rstrip
        if not name.rstrip().lower().endswith(".pdf"):
            continue
        out.append(p)

    # deterministic order
    return sorted(out, key=lambda x: x.name.lower())


def _dest_dir(base: Path, group_by_date: bool) -> Path:
    if group_by_date:
        return base / datetime.now().strftime("%Y%m%d")
    return base


def _safe_move(src: Path, dst_dir: Path) -> Path:
    dst_dir.mkdir(parents=True, exist_ok=True)
    target = dst_dir / src.name
    if target.exists():
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        target = dst_dir / f"{src.stem}_{stamp}{src.suffix}"
    return Path(shutil.move(str(src), str(target)))


def _move_file_group(pdf: Path, dst_dir: Path) -> list[str]:
    """
    Moves a group of files related to a specified PDF to a destination directory. This function moves
    the PDF file itself as well as any associated YAML or YML files found with the same base name.

    :param pdf: The path to the PDF file that needs to be moved. This may also serve as the base
                name for associated YAML or YML files.
    :type pdf: Path
    :param dst_dir: The destination directory where the files will be moved. Must be a valid
                    directory path.
    :type dst_dir: Path
    :return: A list of strings representing the paths of all files successfully moved to the
             destination directory.
    :rtype: list[str]
    """
    moved = []
    if pdf.exists():
        moved.append(str(_safe_move(pdf, dst_dir)))
    for ext in (".yml", ".yaml"):
        side = pdf.with_suffix(ext)
        if side.exists():
            moved.append(str(_safe_move(side, dst_dir)))
    return moved


# ------------------ Sidecar (.yml) handling ------------------
def sidecar_for(pdf: Path) -> Optional[Path]:
    for ext in (".yml", ".yaml"):
        p = pdf.with_suffix(ext)
        if p.exists():
            return p
    return None


def load_sidecar(pdf: Path) -> Dict:
    p = sidecar_for(pdf)
    if not p:
        return {}
    try:
        import yaml
        return yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


# ------------------ Timing utilities ------------------
@contextmanager
def log_timing(logger: logging.Logger, label: str, level=logging.DEBUG):
    t0 = time.perf_counter()
    logger.log(level, f"▶ {label}")
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        logger.log(level, f"✓ {label} (%.2fs)", dt)


# ------------------ State management ------------------
def load_state(state_path: Path) -> Dict:
    if state_path.exists():
        try:
            import json
            return json.loads(state_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"files": {}}


def save_state(st: Dict, state_path: Path):
    import json
    state_path.write_text(json.dumps(st, ensure_ascii=False, indent=2), encoding="utf-8")


def sha256_file(p: Path) -> str:
    """
    Calculate the SHA-256 hash of a file.

    This function reads a file in chunks and calculates its SHA-256 hash.
    It is designed for efficiently handling large files by processing them
    piece by piece.

    :param p: The path to the file for which the SHA-256 hash should be
        calculated.
    :type p: Path
    :return: The hexadecimal SHA-256 hash of the file content.
    :rtype: str
    """
    import hashlib
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------ Directory watching ------------------
def watch_dir(docs_dir: Path, args, ingest_pdf_func, write_report_files_func, update_index_func):
    """
    Watch a directory for PDF files and process them as they appear.

    :param docs_dir: Directory to watch for PDFs
    :param args: Command line arguments
    :param ingest_pdf_func: Function to call for PDF ingestion
    :param write_report_files_func: Function to call for report generation
    :param update_index_func: Function to call for index updates
    """
    logger = logging.getLogger("sense-ingest")
    logger.info("[watch] %s (Ctrl+C to stop)", docs_dir.resolve())

    state_path = Path(getattr(args, 'state_file', '.sense_ingest_state.json'))
    state = load_state(state_path)

    hb = max(getattr(args, "heartbeat", 15), 1)
    # Make the first heartbeat fire immediately
    last_beat = time.time() - hb

    try:
        while True:
            pdfs = _list_pdfs(docs_dir)

            # Debug: show what we see every loop
            logger.debug("Scan %s → %d candidates", docs_dir.resolve(), len(pdfs))
            if pdfs:
                logger.debug("Candidates: %s", ", ".join(p.name for p in pdfs))

            # Idle heartbeat
            now = time.time()
            if not pdfs and now - last_beat >= hb:
                logger.info("… watching (no PDFs). Poll again in %ds", getattr(args, 'poll', 5))
                last_beat = now

            for p in pdfs:
                # Quick unchanged check via size/mtime (fast)
                rec = state["files"].get(str(p))
                try:
                    st = p.stat()
                except FileNotFoundError:
                    continue  # disappeared
                if rec and getattr(args, 'only_new', False) and rec.get("size") == st.st_size and rec.get(
                        "mtime_ns") == st.st_mtime_ns:
                    logger.debug("Skip unchanged (size/mtime): %s", p.name)
                    continue

                # Stable file check so we don't ingest while it's copying
                if not _is_file_stable(p, wait=0.5, checks=4):
                    logger.info("Waiting for file to settle: %s", p.name)
                    continue

                # Real hash for only-new
                try:
                    h = sha256_file(p)
                except Exception as e:
                    logger.warning("Hash failed for %s: %s", p.name, e)
                    continue
                if rec and getattr(args, 'only_new', False) and rec.get("sha256") == h:
                    logger.debug("Skip unchanged (sha256): %s", p.name)
                    # still refresh size/mtime in state
                    state["files"][str(p)]["size"] = st.st_size
                    state["files"][str(p)]["mtime_ns"] = st.st_mtime_ns
                    save_state(state, state_path)
                    continue

                logger.info("[ingest] %s", p.name)
                ok = True
                moved_to: list[str] = []
                final_dir: Optional[Path] = None

                try:
                    stats = ingest_pdf_func(p, args)
                except Exception as e:
                    ok = False
                    logger.exception("Ingest error on %s", p.name)
                    stats = {
                        "file": p.name, "sha256": h, "total": 0, "added": 0, "exists": 0,
                        "per_category": {}, "pages": [], "defaults": {}, "sidecar": False,
                        "rules": getattr(args, "rules", None),
                        "options": {"dry_run": getattr(args, "dry_run", False)},
                        "ok": False, "error": str(e),
                    }

                # Move policy
                if not getattr(args, "dry_run", False) and getattr(args, "move", "success") != "never":
                    try:
                        to_done = ok or (getattr(args, 'move', 'success') == "always")
                        base_dir = Path(getattr(args, 'done_dir', '../docs_done') if to_done else getattr(args, 'fail_dir',
                                                                                                          '../docs_fail'))
                        dst_dir = _dest_dir(base_dir, getattr(args, "group_by_date", False))
                        moved_to = _move_file_group(p, dst_dir)
                        final_dir = dst_dir
                        if moved_to:
                            logger.info("Moved %s → %s", p.name, dst_dir)
                    except Exception as e:
                        logger.warning("Move failed for %s: %s", p.name, e)

                # Reports
                try:
                    rep_dir = Path(getattr(args, "report_dir", "../exports/ingest_reports"))
                    subdir, written = write_report_files_func(rep_dir, stats, moved_to, final_dir,
                                                              getattr(args, "report_format", "md"))
                    if getattr(args, "report_index", False):
                        update_index_func(rep_dir, stats, subdir, written)
                    if written:
                        logger.info("Report: %s", written[-1])
                except Exception as e:
                    logger.warning("Report failed for %s: %s", p.name, e)

                # Persist state (store size/mtime for fast-only-new)
                state["files"][str(p)] = {
                    "sha256": h, "size": st.st_size, "mtime_ns": st.st_mtime_ns,
                    "stats": stats, "ok": ok,
                    "moved_to": moved_to, "final_dir": str(final_dir) if final_dir else None,
                    "ts": int(time.time()),
                }
                save_state(state, state_path)

            time.sleep(getattr(args, "poll", 5))
    except KeyboardInterrupt:
        logger.info("[watch] stopped")




# Enhanced category-specific validation requirements
def get_category_requirements():
    """Return strict requirements for each sensory category."""
    return {
        "smell": {
            "required_words": ["acrid", "putrid", "sweet", "fragrant", "musty", "smoky", "fresh", "stale", "pungent",
                               "aromatic"],
            "required_concepts": ["scent", "odor", "aroma", "stench", "fragrance", "smell", "reek"],
            "description": "Must contain scent/odor vocabulary"
        },
        "sound": {
            "required_words": ["thunderous", "piercing", "soft", "loud", "harsh", "melodic", "distant", "muffled",
                               "sharp", "resonant"],
            "required_concepts": ["roar", "whisper", "scream", "crash", "chime", "rumble", "echo", "clang", "rustle"],
            "description": "Must contain audio quality descriptors"
        },
        "taste": {
            "required_words": ["bitter", "sweet", "sour", "salty", "savory", "metallic", "coppery", "sharp", "mild",
                               "rich"],
            "required_concepts": ["taste", "flavor", "aftertaste"],
            "description": "Must contain taste/flavor vocabulary"
        },
        "touch": {
            "required_words": ["rough", "smooth", "soft", "hard", "hot", "cold", "warm", "cool", "silky", "coarse"],
            "required_concepts": ["texture", "temperature", "pressure", "sensation"],
            "description": "Must contain physical sensation vocabulary"
        },
        "sight": {
            "required_words": ["blazing", "dim", "bright", "brilliant", "pale", "vivid", "gleaming", "shimmering",
                               "glowing", "sparkling"],
            "required_concepts": ["light", "glow", "shine", "reflection", "color intensity"],
            "description": "Must contain visual quality descriptors"
        }
    }



def validate_cultural_authenticity(term: str, category: str, locale: str, era: str, register: str) -> tuple[bool, str]:
    """Simplified cultural authenticity validation."""
    term_lower = term.lower()

    # Check for obvious anachronisms
    anachronisms = {
        "electric", "plastic", "digital", "computer", "phone", "car", "airplane",
        "gunpowder" if era in ["Heian"] else "",
        "tobacco" if era in ["Heian", "Song"] else "",
    }
    anachronisms = {a for a in anachronisms if a}

    if any(ana in term_lower for ana in anachronisms):
        return False, f"anachronistic for {era} period"

    # Allow terms with basic sensory vocabulary
    general_sensory = {
        "smell": ["aromatic", "fragrant", "acrid", "sweet", "smoky", "musty"],
        "sound": ["distant", "soft", "loud", "harsh", "melodic", "thunderous"],
        "taste": ["bitter", "sweet", "sour", "salty", "savory", "rich"],
        "touch": ["rough", "smooth", "soft", "hard", "warm", "cool"],
        "sight": ["bright", "dim", "brilliant", "pale", "gleaming", "shimmering"]
    }

    has_sensory_vocab = any(word in term_lower for word in general_sensory.get(category, []))
    if has_sensory_vocab:
        return True, "general sensory vocabulary"

    return True, "acceptable"


def extract_items(chunk: str, ctx: Dict) -> List[Dict]:
    """Enhanced extraction with validation."""
    logger = logging.getLogger("sense-ingest")

    locale = ctx.get("locale", "")
    era = ctx.get("era", "")
    register = ctx.get("register", "common")

    # Extract using LLM interface (it handles context-aware vs standard internally)
    raw_items = extract_sensory_terms(
        chunk=chunk,
        locale=locale,
        era=era,
        register=register,
        weather=ctx.get("weather", "any")
    )

    # Apply validation logic
    requirements = get_category_requirements()
    validated = []
    rejection_reasons = []

    for item in raw_items:
        term = (item.get("term", "") or "").strip()
        category = (item.get("category", "") or "").strip().lower()
        notes = (item.get("notes", "") or "").strip()

        # Basic validation
        if not term or category not in ALLOWED_CATS:
            continue

        term_lower = term.lower()

        # Apply filters
        if any(banned in term_lower for banned in BAN_TERMS):
            rejection_reasons.append(f"'{term}': banned term")
            continue

        # Emotion/abstract check
        banned_indicators = [
            "annoyed", "angry", "furious", "confident", "nervous", "sad", "happy",
            "glance", "look", "stare", "air", "expression", "manner"
        ]
        if any(banned in term_lower for banned in banned_indicators):
            rejection_reasons.append(f"'{term}': emotion/abstract concept")
            continue

        # Physiological check
        physio_terms = ["breath", "heartbeat", "pulse", "circulation"]
        if any(physio in term_lower for physio in physio_terms):
            rejection_reasons.append(f"'{term}': physiological rather than sensory")
            continue

        # Category-specific validation
        cat_req = requirements[category]
        has_required = (
                any(word in term_lower for word in cat_req["required_words"]) or
                any(concept in term_lower for concept in cat_req["required_concepts"])
        )
        if not has_required:
            rejection_reasons.append(f"'{term}': {cat_req['description']}")
            continue

        # Length check
        if len(term) < MIN_TERM_CHARS:
            rejection_reasons.append(f"'{term}': too short")
            continue

        # Cultural authenticity validation (if we have cultural context)
        if locale and era and register != "common":
            is_authentic, auth_reason = validate_cultural_authenticity(term, category, locale, era, register)
            if not is_authentic:
                rejection_reasons.append(f"'{term}': {auth_reason}")
                continue

        # All validation passed
        validated.append({
            "term": term.title(),
            "category": category,
            "notes": notes
        })

    # Logging
    if locale and era and register != "common":
        logger.info(f"Context-aware extraction for {era} {locale} ({register})")
        logger.info(f"Validated {len(validated)}/{len(raw_items)} culturally authentic terms")
    else:
        logger.info(f"Standard extraction validated {len(validated)}/{len(raw_items)} terms")

    if validated:
        logger.info(f"Valid terms: {', '.join(item['term'] for item in validated[:3])}")
    if rejection_reasons and logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"Filtering: {'; '.join(rejection_reasons[:3])}")

    return validated


def ingest_pdf(pdf_path: Path, args) -> Dict:
    """
    Processes a given PDF file to extract structured information and metadata.

    The function ingests a PDF file, processes its content using OCR (if enabled),
    rules, and other contextual settings. It deduplicates extracted items and
    categorizes them before returning a comprehensive summary, including statistics.

    :param pdf_path: Path to the PDF file to process.
    :type pdf_path: Path
    :param args: Arguments containing configuration options, such as rules,
        language settings, and processing flags.
    :type args: Any
    :return: A dictionary containing the summary of processed items and metadata,
        including the deduplicated items, statistics on their categories, page-level
        summaries, processing options, and ingestion status.
    :rtype: Dict
    """
    t0 = time.perf_counter()
    logger = logging.getLogger("sense-ingest")
    logger.info("▶ Start ingest: %s", pdf_path.name)

    side = load_sidecar(pdf_path)
    rules = load_rules(getattr(args, "rules", None))
    defaults = {
        "locale": side.get("locale") or args.default_locale or "",
        "era": side.get("era") or args.default_era or "",
        "register": side.get("register") or args.default_register or DEFAULT_REGISTER,
        "weather": side.get("weather") or args.default_weather or DEFAULT_WEATHER,
    }
    use_ner = not getattr(args, "no_ner", False)

    t0 = time.perf_counter()
    page_texts = pdf_to_texts(pdf_path, ocr=not args.no_ocr, ocr_lang=args.ocr_lang)
    n_pages = len(page_texts)
    logger.info("Pages: %d (ocr=%s, lang=%s)", n_pages, not args.no_ocr, args.ocr_lang)

    pages_summary: List[Dict] = []
    all_items: List[Dict] = []

    # heartbeat config
    hb = max(getattr(args, "heartbeat", 15), 1)
    last_hb = time.perf_counter()

    for pageno, txt in enumerate(page_texts, 1):
        if not txt.strip():
            logger.debug("Page %d empty, skipping", pageno)
            continue

        # heartbeat
        now = time.perf_counter()
        if now - last_hb >= hb:
            logger.info("… working (%d/%d)", pageno, n_pages)
            last_hb = now

        # infer context
        if "infer_context" in globals():
            ctx = infer_context(txt, rules, defaults, use_ner)
        else:
            ctx = infer_context(txt, rules, defaults)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Page %d ctx: %s", pageno, {k: ctx.get(k) for k in ("locale", "era", "register", "weather")})

        # extract items with timing
        t_llm = time.perf_counter()
        items = extract_items(txt, ctx)
        dt_llm = time.perf_counter() - t_llm
        logger.info("Page %d/%d: %d items (LLM %.2fs)", pageno, n_pages, len(items), dt_llm)
        if logger.isEnabledFor(logging.DEBUG) and items:
            logger.debug("Examples: %s", ", ".join(it["term"] for it in items[:5]))

        # collect rows
        all_items.extend([
            {
                "term": it["term"], "category": it["category"],
                "notes": (it.get("notes", "") + f" (src: {pdf_path.name} p.{pageno})").strip(),
                "locale": ctx.get("locale") or defaults["locale"],
                "era": ctx.get("era") or defaults["era"],
                "register": ctx.get("register") or defaults["register"],
                "weather": ctx.get("weather") or defaults["weather"],
            }
            for it in items
        ])

        # page summary
        by_cat = Counter([it["category"] for it in items])
        pages_summary.append({
            "pageno": pageno,
            "locale": ctx.get("locale") or "",
            "era": ctx.get("era") or "",
            "register": ctx.get("register") or "",
            "weather": ctx.get("weather") or "",
            "items": len(items),
            "examples": [it["term"] for it in items[:5]],
            "by_category": dict(by_cat),
        })

    # dedupe by (term, category, locale, era)
    seen = set()
    uniq: List[Dict] = []
    for rec in all_items:
        key = (rec["term"].lower(), rec["category"], rec["locale"].lower(), rec["era"].lower())
        if key not in seen:
            seen.add(key)
            uniq.append(rec)

    per_cat = Counter([r["category"] for r in uniq])
    logger.info("Deduped to %d unique items (%s)", len(uniq), summarize_cats(dict(per_cat)))

    stats = {
        "file": pdf_path.name,
        "sha256": sha256_file(pdf_path),
        "total": len(uniq),
        "added": 0, "exists": 0,
        "per_category": dict(per_cat),
        "pages": pages_summary,
        "defaults": defaults,
        "sidecar": bool(side),
        "rules": getattr(args, "rules", None),
        "options": {
            "ocr": not args.no_ocr,
            "ocr_lang": args.ocr_lang,
            "ner": use_ner,
            "dry_run": args.dry_run
        },
        "ok": True,
    }

    if args.dry_run:
        logger.info("[dry-run] %s: total=%d (%s)", pdf_path.name, stats["total"],
                    summarize_cats(stats["per_category"]))
        logger.info("✓ Ingest finished in %.2fs (dry-run)", time.perf_counter() - t0)
        return stats

    # write via MCP with progress logs
    logger.info("Writing %d items via MCP…", len(uniq))
    write_stats = {"added": 0, "exists": 0, "error": 0}
    if args.bypass_mcp:
        for rec in uniq:
            res = _csv_add_record_direct(rec)
            st = res.get("status")
            if st == "added":
                write_stats["added"] += 1
            elif st == "exists":
                write_stats["exists"] += 1
            else:
                write_stats["error"] += 1
    else:
        for rec in uniq:
            res = sense_add(rec)  # MCP path
            st = res.get("status")
            if st == "added":
                write_stats["added"] += 1
            elif st == "exists":
                write_stats["exists"] += 1
            else:
                write_stats["error"] += 1
                # optional: log the anomaly
                logging.getLogger("sense-ingest").warning("Write anomaly on %r: %s", rec.get("term"), res)

    stats["added"] = write_stats["added"]
    stats["exists"] = write_stats["exists"]
    stats["write_stats"] = write_stats

    logger.info(
        "✓ Ingest finished in %.2fs (added=%d, exists=%d, errors=%d, total=%d)",
        time.perf_counter() - t0,
        write_stats["added"], write_stats["exists"], write_stats["error"],
        len(uniq),
    )
    return stats


def process_single_pdf_batch(docs_dir: Path, args, ingest_pdf_func, write_report_files_func, update_index_func):
    """
    Process all PDFs in a directory once (batch mode).

    :param docs_dir: Directory containing PDFs to process
    :param args: Command line arguments
    :param ingest_pdf_func: Function to call for PDF ingestion
    :param write_report_files_func: Function to call for report generation
    :param update_index_func: Function to call for index updates
    """
    logger = logging.getLogger("sense-ingest")
    state_path = Path(getattr(args, 'state_file', '.sense_ingest_state.json'))
    state = load_state(state_path)

    for p in sorted(docs_dir.glob("*.pdf")):
        logger.info("[ingest] %s", p.name)
        h = sha256_file(p)
        ok = True
        moved_to: list[str] = []
        final_dir: Optional[Path] = None

        try:
            stats = ingest_pdf_func(p, args)
        except Exception as e:
            ok = False
            logger.exception("Ingest error on %s", p.name)
            stats = {
                "file": p.name, "sha256": h, "total": 0, "added": 0, "exists": 0,
                "per_category": {}, "pages": [], "defaults": {}, "sidecar": False,
                "rules": getattr(args, "rules", None),
                "options": {"dry_run": getattr(args, "dry_run", False)},
                "ok": False, "error": str(e)
            }

        if getattr(args, "print_json", False):
            import json
            print(json.dumps(stats, ensure_ascii=False, indent=2))

        if not getattr(args, "dry_run", False) and getattr(args, "move", "never") != "never":
            to_done = ok or (getattr(args, 'move', 'success') == "always")
            base_dir = Path(
                getattr(args, 'done_dir', '../docs_done') if to_done else getattr(args, 'fail_dir', '../docs_fail'))
            dst_dir = _dest_dir(base_dir, getattr(args, "group_by_date", False))
            moved_to = _move_file_group(p, dst_dir)
            final_dir = dst_dir
            if moved_to:
                logger.info("Moved %s → %s", p.name, dst_dir)

        rep_dir = Path(getattr(args, "report_dir", "../exports/ingest_reports"))
        subdir, written = write_report_files_func(rep_dir, stats, moved_to, final_dir,
                                                  getattr(args, "report_format", "md"))
        if getattr(args, "report_index", False):
            update_index_func(rep_dir, stats, subdir, written)
        if written:
            logger.info("Report: %s", written[-1])

        state["files"][str(p)] = {
            "sha256": h, "stats": stats, "ok": ok,
            "moved_to": moved_to, "final_dir": str(final_dir) if final_dir else None,
            "ts": int(time.time())
        }
        save_state(state, state_path)