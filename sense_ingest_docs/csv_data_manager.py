#!/usr/bin/env python3
"""
CSV and Data Management Functions for Sense-Bank
Handles CSV file operations, data persistence, and record management.
"""

import csv
import os
from pathlib import Path

from typing import Optional, Dict, List


# CSV schema
HEADER = ["term", "category", "locale", "era", "weather", "register", "notes"]

# Optional mapping for nicer filenames per locale
DEFAULT_FILES = {
    "Japan": "jp_sensory.csv",
    "Andes": "andes_sensory.csv",
    "China": "cn_sensory.csv",
    "Venice": "venice_sensory.csv",
    "Abbasid Baghdad": "baghdad_sensory.csv",
    # add more later
}

# Global data directory - can be overridden
DATA_DIR = Path(os.getenv("SENSEBANK_DATA_DIR", Path(__file__).parent / "data"))


def set_data_dir(path: Path) -> None:
    """Set the global data directory for CSV files."""
    global DATA_DIR
    DATA_DIR = path
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def get_data_dir() -> Path:
    """Get the current data directory."""
    return DATA_DIR


def _default_file_for_locale(locale: str) -> str:
    """
    Generate default filename for a given locale.

    :param locale: The locale identifier
    :return: Filename for the locale's CSV file
    """
    key = (locale or "misc").strip()
    return DEFAULT_FILES.get(key, f"{key.lower().replace(' ', '_')}_sensory.csv")


def _csv_load_rows(path: Path) -> List[Dict]:
    """
    Load rows from a CSV file.

    :param path: Path to the CSV file
    :return: List of dictionaries representing CSV rows
    """
    if not path.exists():
        return []

    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [{k: (v or "") for k, v in row.items()} for row in reader]


def _csv_write_rows_atomic(path: Path, rows: List[Dict]) -> None:
    """
    Atomically write rows to a CSV file.
    Creates parent directories if needed and uses temporary file for atomic operation.

    :param path: Path to the CSV file
    :param rows: List of dictionaries to write
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")

    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HEADER)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in HEADER})

    tmp.replace(path)


def _csv_add_record_direct(rec: Dict) -> Dict:
    """
    Client-side add (no MCP). Mirrors sense_add: dedupe by (term,category,locale,era),
    create file if needed, write header if new.

    :param rec: Record dictionary to add
    :return: Dictionary with status information
    """
    record = {
        "term": rec["term"].strip(),
        "category": rec["category"].strip().lower(),
        "locale": rec["locale"].strip(),
        "era": rec["era"].strip(),
        "weather": (rec.get("weather") or "any").strip().lower(),
        "register": (rec.get("register") or "common").strip().lower(),
        "notes": (rec.get("notes") or "").strip(),
    }

    path = DATA_DIR / _default_file_for_locale(record["locale"])
    rows = _csv_load_rows(path)

    # Check for duplicates using composite key
    key = (
        record["term"].lower(),
        record["category"],
        record["locale"].lower(),
        record["era"].lower()
    )

    existing_key = lambda r: (
        r.get("term", "").lower(),
        r.get("category", "").lower(),
        r.get("locale", "").lower(),
        r.get("era", "").lower()
    )

    if any(existing_key(r) == key for r in rows):
        return {"status": "exists", "file": str(path), "record": record}

    rows.append(record)
    _csv_write_rows_atomic(path, rows)
    return {"status": "added", "file": str(path), "record": record}


def load_csv_data(locale: Optional[str] = None) -> List[Dict]:
    """
    Load CSV data for a specific locale or all locales.

    :param locale: Specific locale to load, or None for all
    :return: List of records
    """
    if locale:
        path = DATA_DIR / _default_file_for_locale(locale)
        return _csv_load_rows(path)

    # Load all CSV files
    all_records = []
    for csv_file in DATA_DIR.glob("*.csv"):
        records = _csv_load_rows(csv_file)
        all_records.extend(records)

    return all_records


def get_csv_stats(locale: Optional[str] = None) -> Dict:
    """
    Get statistics about CSV data.

    :param locale: Specific locale to analyze, or None for all
    :return: Dictionary with statistics
    """
    records = load_csv_data(locale)

    # Initialize with empty lists and use sets for collection
    locales_set = set()
    eras_set = set()
    registers_set = set()
    weather_set = set()

    stats = {
        "total_records": len(records),
        "locales": [],
        "eras": [],
        "categories": {},
        "registers": [],
        "weather_conditions": []
    }

    for record in records:
        locales_set.add(record.get("locale", ""))
        eras_set.add(record.get("era", ""))
        registers_set.add(record.get("register", ""))
        weather_set.add(record.get("weather", ""))

        category = record.get("category", "")
        stats["categories"][category] = stats["categories"].get(category, 0) + 1

    # Assign sorted lists
    stats["locales"] = sorted(locales_set)
    stats["eras"] = sorted(eras_set)
    stats["registers"] = sorted(registers_set)
    stats["weather_conditions"] = sorted(weather_set)

    return stats


def backup_csv_files(backup_dir: Path) -> List[str]:
    """
    Create backups of all CSV files.

    :param backup_dir: Directory to store backups
    :return: List of backed up files
    """
    backup_dir.mkdir(parents=True, exist_ok=True)
    backed_up = []

    for csv_file in DATA_DIR.glob("*.csv"):
        backup_path = backup_dir / csv_file.name
        backup_path.write_bytes(csv_file.read_bytes())
        backed_up.append(str(csv_file.name))

    return backed_up


def validate_csv_integrity() -> Dict[str, List[str]]:
    """
    Validate the integrity of CSV files.

    :return: Dictionary with validation results
    """
    issues = {
        "missing_required_fields": [],
        "invalid_categories": [],
        "duplicate_records": [],
        "encoding_issues": []
    }

    valid_categories = {"smell", "sound", "taste", "touch", "sight"}

    for csv_file in DATA_DIR.glob("*.csv"):
        try:
            records = _csv_load_rows(csv_file)
            seen_keys = set()

            for i, record in enumerate(records):
                # Check required fields
                if not record.get("term") or not record.get("category"):
                    issues["missing_required_fields"].append(
                        f"{csv_file.name}:{i + 1} - Missing term or category"
                    )

                # Check valid categories
                category = record.get("category", "").lower()
                if category and category not in valid_categories:
                    issues["invalid_categories"].append(
                        f"{csv_file.name}:{i + 1} - Invalid category: {category}"
                    )

                # Check for duplicates within file
                key = (
                    record.get("term", "").lower(),
                    category,
                    record.get("locale", "").lower(),
                    record.get("era", "").lower()
                )
                if key in seen_keys:
                    issues["duplicate_records"].append(
                        f"{csv_file.name}:{i + 1} - Duplicate record: {record.get('term')}"
                    )
                seen_keys.add(key)

        except UnicodeDecodeError:
            issues["encoding_issues"].append(f"{csv_file.name} - Encoding error")
        except Exception as e:
            issues["encoding_issues"].append(f"{csv_file.name} - Error: {str(e)}")

    return issues


def merge_csv_files(source_files: List[Path], target_file: Path,
                    deduplicate: bool = True) -> Dict:
    """
    Merge multiple CSV files into a single file.

    :param source_files: List of source CSV file paths
    :param target_file: Target merged file path
    :param deduplicate: Whether to remove duplicates
    :return: Dictionary with merge statistics
    """
    all_records = []
    source_stats = {}

    for source_file in source_files:
        if source_file.exists():
            records = _csv_load_rows(source_file)
            all_records.extend(records)
            source_stats[str(source_file)] = len(records)

    original_count = len(all_records)

    if deduplicate:
        seen_keys = set()
        unique_records = []

        for record in all_records:
            key = (
                record.get("term", "").lower(),
                record.get("category", "").lower(),
                record.get("locale", "").lower(),
                record.get("era", "").lower()
            )
            if key not in seen_keys:
                seen_keys.add(key)
                unique_records.append(record)

        all_records = unique_records

    _csv_write_rows_atomic(target_file, all_records)

    return {
        "source_files": source_stats,
        "original_count": original_count,
        "final_count": len(all_records),
        "duplicates_removed": original_count - len(all_records) if deduplicate else 0,
        "target_file": str(target_file)
    }


# Convenience functions for common operations
def add_record(term: str, category: str, locale: str = "", era: str = "",
               weather: str = "any", register: str = "common", notes: str = "") -> Dict:
    """
    Convenience function to add a single record.

    :param term: The sensory term
    :param category: Sensory category (smell, sound, taste, touch, sight)
    :param locale: Geographic/cultural locale
    :param era: Historical era
    :param weather: Weather condition
    :param register: Language register
    :param notes: Additional notes
    :return: Result dictionary
    """
    record = {
        "term": term,
        "category": category,
        "locale": locale,
        "era": era,
        "weather": weather,
        "register": register,
        "notes": notes
    }
    return _csv_add_record_direct(record)


def search_records(term_pattern: str = "", category: str = "",
                   locale: str = "", era: str = "") -> List[Dict]:
    """
    Search for records matching given criteria.

    :param term_pattern: Pattern to match in terms (case-insensitive)
    :param category: Exact category match
    :param locale: Exact locale match
    :param era: Exact era match
    :return: List of matching records
    """
    all_records = load_csv_data()
    matching_records = []

    for record in all_records:
        # Check term pattern
        if term_pattern and term_pattern.lower() not in record.get("term", "").lower():
            continue

        # Check exact matches
        if category and record.get("category", "") != category:
            continue
        if locale and record.get("locale", "") != locale:
            continue
        if era and record.get("era", "") != era:
            continue

        matching_records.append(record)

    return matching_records