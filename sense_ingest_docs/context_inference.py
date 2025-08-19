#!/usr/bin/env python3
"""
Context Inference Module

This module provides functionality for inferring contextual information from text,
including locale, era, register, and weather conditions. It uses rule-based matching,
named entity recognition (NER), and heuristic analysis.
"""

import re
import sys
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Optional


def ner_hints(text: str) -> Dict[str, List[str]]:
    """
    Extracts named entities of specific types from a given text and organizes them into
    a dictionary. This function uses the spaCy library to process the text and identify
    entities categorized as "GPE", "LOC", and "DATE". If spaCy is not installed or an
    error occurs during execution, the function gracefully handles the exception and
    returns an empty dictionary for the specified keys.

    :param text: Input text to extract named entities from.
    :type text: str

    :return: A dictionary with entity types ("GPE", "LOC", "DATE") as keys and lists of
        corresponding entities as values.
    :rtype: Dict[str, List[str]]
    """
    out = {"GPE": [], "LOC": [], "DATE": []}
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text)
        for ent in doc.ents:
            if ent.label_ in out:
                out[ent.label_].append(ent.text)
    except Exception:
        pass
    return out


def load_rules(path: Optional[str]) -> Dict:
    """
    Load rules from a YAML file located at the specified path.

    This function reads YAML configuration data from the file at the given path.
    If the path is not provided or the file does not exist, an empty dictionary
    is returned. If there is an error during the file-reading or parsing process,
    a warning message is printed to standard error, and an empty dictionary is returned.

    :param path: Optional file path to the YAML configuration file.
    :type path: Optional[str]
    :return: A dictionary containing the configuration rules loaded from the file,
        or an empty dictionary if the path is not specified, the file does not exist,
        or an error occurs while reading/parsing the file.
    :rtype: Dict
    """
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        print(f"[WARN] rules file not found: {p}", file=sys.stderr)
        return {}
    try:
        return yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except Exception as e:
        print(f"[WARN] failed to read rules: {e}", file=sys.stderr)
        return {}


def apply_rules(text: str, rules: Dict) -> Dict:
    """
    Applies a set of matching rules to determine inferred categories from the given text.

    The function processes the input text and applies a series of regular expression-based
    rules provided in the `rules` dictionary. These rules are used to identify and label
    certain attributes, such as "locale," "era," "register," or "weather," in the text.
    The results are stored in a dictionary, mapping specific keys to inferred labels.

    :param text: The input text to be processed and matched against the rules.
    :type text: str
    :param rules: A dictionary containing sets of label-to-pattern mapping for various
        categories (e.g., locales, eras, registers, etc.) to guide the matching logic.
    :type rules: Dict
    :return: A dictionary containing inferred category labels and their corresponding
        matching results based on the input rules.
    :rtype: Dict
    """
    inferred = {}

    def match_block(block: Dict, target_key: str):
        if not block:
            return
        for label, pats in block.items():
            for pat in pats or []:
                try:
                    if re.search(pat, text, flags=re.I):
                        inferred[target_key] = label
                        return
                except re.error:
                    continue

    match_block(rules.get("locales"), "locale")
    match_block(rules.get("eras"), "era")
    match_block(rules.get("registers"), "register")
    match_block(rules.get("weather"), "weather")
    return inferred


def infer_context(page_text: str, rules: dict | None, defaults: dict, use_ner: bool = True) -> dict:
    """
    Infer {locale, era, register, weather} from a page of text.
    - Start with defaults
    - Apply YAML rules if available (safe fallback)
    - If locale/era still missing and use_ner=True, try NER hints
    - Fill remaining gaps with simple heuristics
    """
    logger = logging.getLogger("sense-ingest")

    t = page_text or ""
    ctx = {
        "locale": (defaults.get("locale") or "").strip(),
        "era": (defaults.get("era") or "").strip(),
        "register": (defaults.get("register") or "common").strip(),
        "weather": (defaults.get("weather") or "any").strip(),
    }
    sources: list[str] = ["defaults"]

    # ---------- 1) Rules (with safe fallback) ----------
    def _apply_rules_fallback(text: str, rulebook: dict | None) -> dict:
        out: dict = {}
        if not rulebook:
            return out

        def _pick(section: str, key: str):
            section_map = rulebook.get(section, {}) or {}
            for label, patterns in section_map.items():
                for pat in (patterns or []):
                    try:
                        if re.search(pat, text, re.I):
                            out[key] = label
                            return
                    except re.error:
                        # Ignore malformed regex in YAML
                        continue

        _pick("locales", "locale")
        _pick("eras", "era")
        _pick("registers", "register")
        _pick("weather", "weather")
        return out

    inferred = {}
    try:
        if rules:
            # If your project provides apply_rules, use it:
            inferred = apply_rules(t, rules)
    except Exception:
        inferred = _apply_rules_fallback(t, rules)

    if inferred:
        for k, v in inferred.items():
            if v:
                ctx[k] = str(v).strip()
        sources.append("rules")

    # ---------- 2) NER hints (only if locale/era still missing) ----------
    if use_ner and (not ctx["locale"] or not ctx["era"]):
        try:
            hints = ner_hints(t[:2000])
        except Exception:
            hints = {}

        gpes = " ".join(hints.get("GPE", []))
        dates = " ".join(hints.get("DATE", []))
        hay = f"{t} {gpes} {dates}"

        before_loc, before_era = ctx["locale"], ctx["era"]

        if not ctx["locale"]:
            if re.search(r"\b(Kyoto|Kyōto|Nara|Heian|Yamato|Japan)\b", hay, re.I):
                ctx["locale"] = "Japan"
            elif re.search(r"\b(Baghdad|Tigris|Samarra)\b", hay, re.I):
                ctx["locale"] = "Abbasid Baghdad"
            elif re.search(r"\b(Venice|Venezia|Rialto|Lagoon)\b", hay, re.I):
                ctx["locale"] = "Venice"
            elif re.search(r"\b(Cusco|Cuzco|Andes|Quito|Titicaca|Chim[úu]|Moche)\b", hay, re.I):
                ctx["locale"] = "Andes"

        if not ctx["era"]:
            if re.search(r"\bHeian\b|\b(79[4-9]|8\d\d|11[0-7]\d)\b", hay, re.I):
                ctx["era"] = "Heian"
            elif re.search(r"\bSong\b", hay, re.I):
                ctx["era"] = "Song"
            elif re.search(r"\bMughal\b", hay, re.I):
                ctx["era"] = "Mughal"
            elif re.search(r"\bChim[úu]\b|\bChim[úu]\s*Empire\b", hay, re.I):
                ctx["era"] = "Chimú"

        if ctx["locale"] != before_loc or ctx["era"] != before_era:
            sources.append("ner")

    # ---------- 3) Heuristics for register/weather ----------
    tl = t.lower()
    before_reg, before_wx = ctx["register"], ctx["weather"]

    if not ctx["register"] or ctx["register"].lower() == "common":
        if re.search(r"\b(palace|chamberlain|courtier|imperial|empress|retainers?)\b", tl):
            ctx["register"] = "court"
        elif re.search(r"\b(quay|rigging|tar|harbour|harbor|deck|keel|mast|bilge|sail)\b", tl):
            ctx["register"] = "maritime"
        elif re.search(r"\b(monastery|temple|abbot|sutra|monk|bell|cloister)\b", tl):
            ctx["register"] = "monastic"
        elif re.search(r"\b(festival|procession|drum|lantern|bonfire|parade)\b", tl):
            ctx["register"] = "festival"
        elif re.search(r"\b(market|bazaar|vendor|stall|hawker|merchant)\b", tl):
            ctx["register"] = "market"
        elif re.search(r"\b(garrison|encampment|spear|banner|shield|lance|archer)\b", tl):
            ctx["register"] = "military"
        elif re.search(r"\b(scroll|inkstone|calligraphy|study|classic|scholar)\b", tl):
            ctx["register"] = "scholar"
        elif re.search(r"\b(garden|moss|gravel|raked|lantern|pond|koi)\b", tl):
            ctx["register"] = "garden"

    if not ctx["weather"] or ctx["weather"].lower() == "any":
        if re.search(r"\bmonsoon\b", tl):
            ctx["weather"] = "monsoon"
        elif re.search(r"\bdry\s+season\b", tl):
            ctx["weather"] = "dry season"
        elif re.search(r"\bwinter\s+snow\b|\bsnowfall\b|\bsnow\b", tl):
            ctx["weather"] = "winter snow"
        elif re.search(r"\brain(fall)?|downpour|storm|shower|drizzle\b", tl):
            ctx["weather"] = "rain"
        elif re.search(r"\bdawn|daybreak|sunrise\b", tl):
            ctx["weather"] = "dawn"
        elif re.search(r"\bnight|moonlit|midnight|nightfall\b", tl):
            ctx["weather"] = "night"
        elif re.search(r"\bsummer\b", tl):
            ctx["weather"] = "summer"
        elif re.search(r"\bwinter\b", tl):
            ctx["weather"] = "winter"

    if ctx["register"] != before_reg or ctx["weather"] != before_wx:
        sources.append("heuristics")

    # ---------- 4) Normalize enums ----------
    def norm_space(s: str) -> str:
        return " ".join((s or "").split())

    ctx["locale"] = norm_space(ctx["locale"])
    ctx["era"] = norm_space(ctx["era"])
    ctx["register"] = (norm_space(ctx["register"]) or "common").lower()
    ctx["weather"] = (norm_space(ctx["weather"]) or "any").lower()

    logger.debug("Context %s via %s", ctx, ", ".join(sources))
    return ctx