#!/usr/bin/env python3

import logging

# Import PDF processing and watching functionality
from sense_ingest_docs.pdf_watcher import (
    watch_dir, process_single_pdf_batch, DEFAULT_REGISTER, DEFAULT_WEATHER, ingest_pdf
)

from sense_ingest_docs.reporting import update_index

# Import CSV and data management functions
from sense_ingest_docs.csv_data_manager import (
    set_data_dir,
)

def setup_logging(level: str = "INFO", log_file: str | None = None) -> logging.Logger:
    logger = logging.getLogger("sense-ingest")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()

    fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(message)s", "%H:%M:%S")

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if log_file:
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    # quiet some noisy libs
    logging.getLogger("pdfplumber").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("pdf2image").setLevel(logging.WARNING)
    return logger


def main(write_report_files=None):
    """CLI entrypoint for ingesting PDFs and writing Sense-Bank rows + reports."""
    import argparse
    from pathlib import Path

    ap = argparse.ArgumentParser(
        description="Ingest PDFs from docs/ and add sensory cues to Sense-Bank via MCP."
    )
    ap.add_argument("--docs-dir", default="docs")
    ap.add_argument("--rules")
    ap.add_argument("--default-locale")
    ap.add_argument("--default-era")
    ap.add_argument("--default-register", default=DEFAULT_REGISTER)
    ap.add_argument("--default-weather", default=DEFAULT_WEATHER)
    ap.add_argument("--no-ocr", action="store_true")
    ap.add_argument("--ocr-lang", default="eng")
    ap.add_argument("--data-dir", default="data", help="Directory for Sense-Bank CSVs")
    ap.add_argument("--state-file", default="exports/ingest_reports/state.json")
    ap.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    ap.add_argument("--log-file", default=None)
    ap.add_argument("--heartbeat", type=int, default=15)
    ap.add_argument("--trace-llm", action="store_true")
    ap.add_argument("--done-dir", default="docs_done")
    ap.add_argument("--fail-dir", default="docs_fail")
    ap.add_argument("--move", choices=["success", "always", "never"], default="success")
    ap.add_argument("--bypass-mcp", action="store_true",
                    help="Write directly to CSVs instead of calling MCP")
    ap.add_argument("--report-dir", default="exports/ingest_reports")
    ap.add_argument("--report-format", choices=["md", "json", "both"], default="md")
    ap.add_argument("--report-index", action="store_true")
    ap.add_argument("--group-by-date", action="store_true")
    ap.add_argument("--print-json", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--only-new", action="store_true")
    ap.add_argument("--poll", type=int, default=5)
    args = ap.parse_args()

    # Set the global data directory using the CSV manager
    set_data_dir(Path(args.data_dir))

    # initialize logging for everything below
    setup_logging(args.log_level, args.log_file)
    logger = logging.getLogger("sense-ingest")

    docs_dir = Path(args.docs_dir)
    docs_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Sense-Bank Ingest — watch=%s, once=%s, docs=%s, bypass_mcp=%s",
        not args.once, args.once, docs_dir, args.bypass_mcp
    )

    if args.bypass_mcp:
        logger.warning("BYPASS MCP is ON — writing directly to CSVs.")
    else:
        logger.info("MCP mode — writing via server tools (sense_add, etc.).")

    # Import and use the actual function if none provided
    if write_report_files is None:
        from sense_ingest_docs.reporting import write_report_files

    if args.once:
        process_single_pdf_batch(docs_dir, args, ingest_pdf, write_report_files, update_index)
        return

    # watch mode
    watch_dir(docs_dir, args, ingest_pdf, write_report_files, update_index)


if __name__ == "__main__":
    main()