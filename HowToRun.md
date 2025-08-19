# Sense‑Bank Local Ingestion Runbook

This guide shows you how to run the **Sense‑Bank** stack locally on your laptop: the MCP server, the Strands chat, the text reader agent, and the PDF ingester with reports and auto‑move. Everything runs offline with **Ollama** using an OpenAI‑compatible API.

---

## 0) Prereqs

* **macOS** with Python ≥ 3.12 (you’re using `~/venvs/strands`).
* **Ollama** installed and running (`ollama serve`).
* Pull a local model (example):

  ```bash
  ollama pull llama3.1:8b
  ```
* Homebrew tools for OCR (optional, only for image‑only PDFs):

  ```bash
  brew install tesseract poppler
  ```
* Python libs (activate your venv first):

  ```bash
  source ~/venvs/strands/bin/activate
  pip install -r requirements.txt
  pip install pdfplumber pdf2image pytesseract pillow pyyaml requests xlsxwriter spacy
  python -m spacy download en_core_web_sm
  ```

  > If you see NumPy/Torch warnings, you can usually ignore them or uninstall torch in this venv: `pip uninstall -y torch torchvision torchaudio`.

---

## 1) Repo layout (key files)

```
sense-bank-starter/
├─ sense_bank/                 # library code
├─ data/                       # CSV corpora (auto-created per locale)
├─ memory/                     # agent memory (used_terms, etc.)
├─ exports/                    # CSV/MD/XLSX exports + ingest reports
├─ docs/                       # drop PDFs here for ingestion
├─ mcp_sense_bank.py           # MCP server (FastMCP)
├─ strands_chat.py             # interactive chat loop
├─ sense_reader_agent.py       # (optional) text/markdown extractor
├─ sense_ingest_docs.py        # PDF ingester (watch/once, OCR, reports, moves)
├─ ingest_rules.yaml           # (optional) regex rules for locale/era/register
└─ requirements.txt
```

---

## 2) Environment

Set these in each terminal before running a component:

```bash
source ~/venvs/strands/bin/activate
export OPENAI_API_BASE=http://localhost:11434/v1
export OPENAI_API_KEY=ollama
export SENSEBANK_MODEL=llama3.1:8b
export MCP_URL=http://127.0.0.1:8000/mcp/
```

---

## 3) Start the MCP server

In Terminal A:

```bash
cd /path/to/sense-bank-starter
python mcp_sense_bank.py
# serves FastMCP at http://127.0.0.1:8000/mcp/
```

You’ll see `Uvicorn running on http://127.0.0.1:8000` and occasional `307 Temporary Redirect` logs (expected for the streamable HTTP transport).

---

## 4) Use the Strands chat (interactive)

In Terminal B:

```bash
cd /path/to/sense-bank-starter
python strands_chat.py
```

**Cheat‑sheet (selected):**

```
/set key=val ...                 # e.g. /set locale=Japan era=Heian register=court weather=rain n=8 no_repeat=true
/suggest [key=val ...]           # pulls cues (supports exclude=smell,sound exclude_terms="a,b" no_repeat=true)
/rewrite <your paragraph>        # rewrites with selected cues
/list [filters]                  # browse corpus
/add /edit /delete               # maintain corpus rows
/csv|/md|/xlsx [filters]         # export packs
/pack [filters]                  # write CSV+MD with one basename
/importmd path=...               # import MD table back into CSVs
/search q="..."                 # fuzzy search terms/notes
/validate [options]              # normalize, dedupe, enforce enums
/help                            # print all commands
```

Example session:

```text
/set locale=Japan era=Heian register=court weather=rain n=8 no_repeat=true
/suggest exclude=smell
/rewrite After the storm, she paused beneath the eaves…
```

---

## 5) (Optional) Run the text reader agent

Extract cues from a `.txt` / `.md` and write to Sense‑Bank via MCP.

```bash
python sense_reader_agent.py --file excerpt.txt \
  --locale "Japan" --era Heian --register court --weather rain \
  --print-json --dry-run
# then without --dry-run to write
```

---

## 6) Run the PDF ingester

Place PDFs in `docs/`. Optionally add a sidecar next to each PDF (e.g., `novel.yml`):

```yaml
locale: Japan
era: Heian
register: court
weather: rain
```

Add/extend regex rules in `ingest_rules.yaml` for auto‑inference.

### One‑shot dry run

```bash
python sense_ingest_docs.py --docs-dir docs --rules ingest_rules.yaml --once --dry-run
```

### One‑shot write + move processed files + write reports

```bash
python sense_ingest_docs.py --docs-dir docs --rules ingest_rules.yaml --once \
  --move success --done-dir docs_done --fail-dir docs_fail --group-by-date \
  --report-dir exports/ingest_reports --report-format md --report-index
```

### Watch mode (polls for new PDFs)

```bash
python sense_ingest_docs.py --docs-dir docs --rules ingest_rules.yaml --only-new \
  --move success --done-dir docs_done --fail-dir docs_fail --group-by-date \
  --report-dir exports/ingest_reports --report-format both --report-index
```

**Useful flags:**

* `--no-ocr` to skip OCR on image‑only PDFs (faster).
* `--ocr-lang` (default `eng`) for Tesseract language.
* `--no-ner` to disable spaCy NER hints.
* `--default-locale/--default-era` for batch defaults.
* `--target-file` (in reader agent) to force a specific CSV output file.

**How context is inferred (per page):** sidecar → regex rules → spaCy NER hints → CLI defaults. The chosen context is passed to the LLM extractor.

---

## 7) Reports & where files go

* Per‑file reports land under `exports/ingest_reports/<pdf-stem>/`:

  * `latest.md` (and/or `latest.json`)
  * timestamped snapshots like `20250818_142212.md`
* A rolling `exports/ingest_reports/index.md` links to the latest report for each PDF.
* After ingest, source files move to:

  * **done** (e.g., `docs_done/YYYYMMDD/...`) on success (or if `--move always`)
  * **fail** on error. Sidecar `.yml/.yaml` moves with the PDF.

Each report summarizes:

* totals: `total`, `added`, `exists`
* per‑category histogram (smell/sound/taste/touch/sight)
* inferred context per page (locale/era/register/weather)
* example terms
* options used (OCR/NER/dry‑run)
* final move location

---

## 8) Troubleshooting

* **Model not found** (Ollama): `ollama pull llama3.1:8b`; ensure `OPENAI_API_BASE=http://localhost:11434/v1`.
* **NumPy / Torch warnings**: safe to ignore for this workflow; or uninstall torch in this venv.
* **OCR slow**: run with `--no-ocr` or install `tesseract`/`poppler` via Homebrew.
* **Incorrect locale/era**: add/adjust regexes in `ingest_rules.yaml` or provide a per‑PDF sidecar.
* **Duplicates**: run `/validate dedupe=true` in the chat after big ingests.

---

## 9) Sidecar & rules examples

**Sidecar `novel.yml`**

```yaml
locale: Japan
era: Heian
register: court
weather: rain
```

**Rules `ingest_rules.yaml`**

```yaml
locales:
  Japan: ["\\bHeian|Kyoto|Nara|Yamato\\b", "\\bGenji\\b"]
eras:
  Heian: ["\\bHeian\\b", "\\b(79[4-9]|8\\d\\d|11[0-7]\\d)\\b"]
registers:
  court: ["\\bcourt\\b", "\\bpalace\\b"]
weather:
  rain: ["\\brain|storm|eaves\\b"]
```

---

## 10) Daily workflow template (30–60 min)

1. **Drop PDFs** into `docs/` (add sidecars if you know the context).
2. Run ingester once with `--dry-run` → review `exports/ingest_reports/*/latest.md`.
3. Re‑run without `--dry-run` to write + move + report.
4. In chat: `/validate normalize_case=true fix_whitespace=true dedupe=true dry_run=false`.
5. Export a scene pack: `/pack locale=Japan era=Heian register=court`.
6. Draft with freshness: `/set no_repeat=true exclude=smell` → `/rewrite ...`.

---

**That’s it.** If you want this packaged as a Makefile (`make serve`, `make ingest`, `make pack`), say the word and we’ll add it.
