#!/usr/bin/env python3
"""
Reporting and analysis functionality for PDF ingestion.
Handles report generation, statistics, and file organization.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple


# ------------------ Utility Functions ------------------
def _ensure_dir(path: Path):
    """Ensure directory exists, creating it if necessary."""
    path.mkdir(parents=True, exist_ok=True)


def _safe_rel(parent: Path, child: Path) -> str:
    """Get the relative path safely, falling back to absolute if needed."""
    try:
        return str(child.relative_to(parent))
    except Exception:
        return str(child)


def summarize_cats(d: Dict[str, int]) -> str:
    """
    Summarizes the given categories in a predefined order and includes any additional
    categories not defined in the order. The resulting summary is a comma-separated
    string with each category and its associated value.

    :param d: Dictionary of categories (str) with their associated integer values.
    :type d: Dict[str, int]
    :return: A string summarizing the categories and their values.
    :rtype: str
    """
    if not d:
        return "-"

    order = ["smell", "sound", "taste", "touch", "sight"]
    parts = [f"{k}:{d.get(k, 0)}" for k in order if k in d or d.get(k, 0)]

    # add any other stray categories
    for k, v in d.items():
        if k not in order:
            parts.append(f"{k}:{v}")

    return ", ".join(parts) if parts else "-"


# ------------------ Report Generation ------------------
def render_report_md(stats: Dict, moved_to: List[str], final_dir: Optional[Path]) -> str:
    """
    Generates a Markdown string representing a detailed report of the ingest statistics.

    The function processes given statistics, including file details, processing statuses,
    defaults, and options. It also includes a summary of categorized data, page-specific
    information, and directories processed during the operation.

    :param stats: A dictionary containing ingest statistics. The expected keys include:
                  - 'file': The name of the file being processed.
                  - 'sha256': Optional checksum of the file.
                  - 'total': Total number of records processed.
                  - 'added': Number of records added.
                  - 'exists': Number of records that already exist.
                  - 'defaults': A dictionary of default metadata like locale, era, register,
                    and weather.
                  - 'per_category': A dictionary summarizing totals by category.
                  - 'options': A dictionary of process options such as OCR, dry run status,
                    and named entity recognition.
                  - 'sidecar': A boolean indicating if a sidecar file is present.
                  - 'rules': An optional string indicating a rules file applied.
                  - 'ok': A boolean indicating if the operation succeeded.
                  - 'pages': A list of page-specific details, such as:
                    - 'pageno': Page number.
                    - 'locale', 'era', 'register', 'weather': Metadata for each page.
                    - 'items': Total items found on the page.
                    - 'examples': List of examples found on the page (up to 5 shown).
                    - 'by_category': Summary totals by category per page.
    :param moved_to: A list of directories or locations where files were moved.
    :param final_dir: An optional Path object representing the final directory's location.

    :return: A string in markdown format representing the ingest report.
    """
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    per_cat = summarize_cats(stats.get("per_category", {}))
    defaults = stats.get("defaults", {})
    options = stats.get("options", {})
    sidecar = stats.get("sidecar", False)
    rules = stats.get("rules")
    ok = stats.get("ok", True)
    status = "ok" if ok and not options.get("dry_run") else ("dry-run" if options.get("dry_run") else "error")

    lines = []
    lines.append(f"# Ingest Report — {stats['file']}")
    lines.append("")
    lines.append(f"- **Generated:** {stamp}")
    lines.append(f"- **SHA256:** `{stats.get('sha256', '')}`")
    lines.append(f"- **Status:** **{status}**")
    lines.append(
        f"- **Totals:** total={stats.get('total', 0)}, added={stats.get('added', 0)}, exists={stats.get('exists', 0)}")
    lines.append(f"- **By category:** {per_cat}")
    lines.append(
        f"- **Defaults:** locale={defaults.get('locale', '')}, era={defaults.get('era', '')}, register={defaults.get('register', '')}, weather={defaults.get('weather', '')}")
    lines.append(f"- **Sidecar present:** {bool(sidecar)}")
    lines.append(f"- **Rules file:** {rules or '-'}")
    lines.append(
        f"- **Options:** ocr={options.get('ocr')}, ocr_lang={options.get('ocr_lang')}, ner={options.get('ner', None)}, dry_run={options.get('dry_run')}")

    if moved_to:
        lines.append(f"- **Moved to:**")
        for m in moved_to:
            lines.append(f"  - `{m}`")

    if final_dir:
        lines.append(f"- **Final directory:** `{final_dir}`")

    lines.append("")

    # Pages table
    pages = stats.get("pages", [])
    if pages:
        lines.append("## Pages")
        lines.append("| page | locale | era | register | weather | items | examples | by_category |")
        lines.append("|---:|---|---|---|---|---:|---|---|")
        for pg in pages:
            ex = ", ".join(pg.get("examples", [])[:5])
            bc = summarize_cats(pg.get("by_category", {}))
            lines.append(
                f"| {pg.get('pageno', '')} | {pg.get('locale', '')} | {pg.get('era', '')} | {pg.get('register', '')} | {pg.get('weather', '')} | {pg.get('items', 0)} | {ex} | {bc} |")
        lines.append("")

    return "\n".join(lines)


def write_report_files(report_dir: Path, stats: Dict, moved_to: List[str], final_dir: Optional[Path], fmt: str) -> \
Tuple[Path, List[Path]]:
    """
    Writes report files in specified formats (Markdown, JSON, or both) to a specified subdirectory.
    Each generated report file is tagged with a timestamp. Optionally, pointers to the most recently
    generated reports are written for quick access.

    The function creates the necessary subdirectories before writing the files to ensure
    proper file organization.

    :param report_dir: The base directory to store the generated report files.
    :type report_dir: Path
    :param stats: The statistics or data to include in the reports.
    :type stats: Dict
    :param moved_to: List of filenames or paths that the process moved during its execution.
    :type moved_to: List[str]
    :param final_dir: The final directory where the process output resides. Can be None.
    :type final_dir: Optional[Path]
    :param fmt: The format(s) of the report to generate. Accepts 'md', 'json', or 'both'.
    :type fmt: str
    :return: A tuple containing the subdirectory where reports are stored and a list of generated file paths.
    :rtype: Tuple[Path, List[Path]]
    """
    _ensure_dir(report_dir)
    stem = Path(stats["file"]).stem
    subdir = report_dir / stem
    _ensure_dir(subdir)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    written = []

    if fmt in ("md", "both"):
        md = render_report_md(stats, moved_to, final_dir)
        md_path = subdir / f"{ts}.md"
        md_path.write_text(md, encoding="utf-8")
        # keep a 'latest.md' pointer
        (subdir / "latest.md").write_text(md, encoding="utf-8")
        written.append(md_path)

    if fmt in ("json", "both"):
        js = dict(stats)
        js["moved_to"] = moved_to
        js["final_dir"] = str(final_dir) if final_dir else None
        js["generated_at"] = datetime.now().isoformat()
        js_path = subdir / f"{ts}.json"
        js_path.write_text(json.dumps(js, ensure_ascii=False, indent=2), encoding="utf-8")
        (subdir / "latest.json").write_text(json.dumps(js, ensure_ascii=False, indent=2), encoding="utf-8")
        written.append(js_path)

    return subdir, written


def update_index(report_dir: Path, stats: Dict, subdir: Path, written: List[Path]):
    """
    Updates the index file in the given report directory with statistics about
    the current processing and optionally a link to the latest report in the subdirectory.

    The method ensures that the directory exists, writes a new table header if the
    index file does not already exist, and appends a formatted line of statistics
    with links to relevant reports. This functionality is essential for maintaining an
    overview of processing results.

    :param report_dir: The directory where the index file should be updated.
    :type report_dir: Path
    :param stats: A dictionary containing statistics about the operation
                  (e.g., file name, total, added, exists, processing status).
    :type stats: Dict
    :param subdir: The subdirectory containing processed files and reports.
    :type subdir: Path
    :param written: A list of paths of written report files during the current operation.
    :type written: List[Path]
    :return: None
    """
    idx = report_dir / "index.md"
    _ensure_dir(report_dir)

    if not idx.exists():
        idx.write_text(
            "# Ingest Index\n\n| date | file | status | total | added | exists | by_category | report |\n|---|---|---|---:|---:|---:|---|---|\n",
            encoding="utf-8")

    date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    status = "ok" if stats.get("ok", True) and not stats.get("options", {}).get("dry_run") else (
        "dry-run" if stats.get("options", {}).get("dry_run") else "error")
    cats = summarize_cats(stats.get("per_category", {}))

    # link to latest.md (exists for md/both), else first written file
    link_target = subdir / ("latest.md" if (subdir / "latest.md").exists() else (written[0].name if written else ""))
    link = _safe_rel(report_dir, link_target)

    line = f"| {date} | {stats['file']} | {status} | {stats.get('total', 0)} | {stats.get('added', 0)} | {stats.get('exists', 0)} | {cats} | [{Path(link).name}]({link}) |\n"

    with idx.open("a", encoding="utf-8") as f:
        f.write(line)


# ------------------ Analysis Functions ------------------
def analyze_extraction_quality(stats: Dict) -> Dict:
    """
    Analyze the quality and patterns in extraction results.

    :param stats: Statistics dictionary from PDF processing
    :return: Analysis results dictionary
    """
    analysis = {
        "total_items": stats.get("total", 0),
        "success_rate": 0.0,
        "category_distribution": {},
        "pages_processed": len(stats.get("pages", [])),
        "avg_items_per_page": 0.0,
        "quality_score": 0.0
    }

    # Calculate success rate
    total = stats.get("total", 0)
    added = stats.get("added", 0)
    if total > 0:
        analysis["success_rate"] = (added / total) * 100

    # Category distribution
    per_category = stats.get("per_category", {})
    if per_category:
        total_cats = sum(per_category.values())
        analysis["category_distribution"] = {
            cat: (count / total_cats) * 100
            for cat, count in per_category.items()
        }

    # Average items per page
    pages = stats.get("pages", [])
    if pages:
        total_page_items = sum(pg.get("items", 0) for pg in pages)
        analysis["avg_items_per_page"] = total_page_items / len(pages)

    # Simple quality score based on various factors
    quality_factors = []

    # Factor 1: Category diversity (more categories = better)
    if len(per_category) >= 4:
        quality_factors.append(10)
    elif len(per_category) >= 2:
        quality_factors.append(7)
    else:
        quality_factors.append(3)

    # Factor 2: Items per page ratio (sweet spot around 3-8 items per page)
    avg_items = analysis["avg_items_per_page"]
    if 3 <= avg_items <= 8:
        quality_factors.append(10)
    elif 1 <= avg_items <= 12:
        quality_factors.append(7)
    else:
        quality_factors.append(4)

    # Factor 3: Success rate
    success_rate = analysis["success_rate"]
    if success_rate >= 80:
        quality_factors.append(10)
    elif success_rate >= 60:
        quality_factors.append(7)
    else:
        quality_factors.append(4)

    analysis["quality_score"] = sum(quality_factors) / len(quality_factors)

    return analysis


def generate_summary_stats(all_stats: List[Dict]) -> Dict:
    """
    Generate summary statistics across multiple processing runs.

    :param all_stats: List of statistics dictionaries from multiple PDF processing runs
    :return: Summary statistics dictionary
    """
    if not all_stats:
        return {"error": "No statistics provided"}

    summary = {
        "total_files": len(all_stats),
        "successful_files": len([s for s in all_stats if s.get("ok", False)]),
        "total_items_extracted": sum(s.get("total", 0) for s in all_stats),
        "total_items_added": sum(s.get("added", 0) for s in all_stats),
        "total_items_existing": sum(s.get("exists", 0) for s in all_stats),
        "avg_items_per_file": 0.0,
        "category_totals": {},
        "processing_times": [],
        "quality_scores": []
    }

    # Calculate averages
    if summary["total_files"] > 0:
        summary["avg_items_per_file"] = summary["total_items_extracted"] / summary["total_files"]

    # Aggregate category totals
    all_categories = {}
    for stats in all_stats:
        per_cat = stats.get("per_category", {})
        for cat, count in per_cat.items():
            all_categories[cat] = all_categories.get(cat, 0) + count
    summary["category_totals"] = all_categories

    # Collect quality scores
    for stats in all_stats:
        analysis = analyze_extraction_quality(stats)
        summary["quality_scores"].append(analysis["quality_score"])

    if summary["quality_scores"]:
        summary["avg_quality_score"] = sum(summary["quality_scores"]) / len(summary["quality_scores"])

    return summary