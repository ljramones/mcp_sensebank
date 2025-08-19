# Sense Ingestor

A sophisticated PDF processing system that extracts culturally-contextualized sensory descriptions from historical documents using AI-powered text analysis.

## Overview

The Sense Ingestor automatically monitors a directory for PDF documents, processes them page-by-page through a Large Language Model (LLM), and extracts high-quality sensory descriptions (smell, sound, taste, touch, sight) while maintaining historical and cultural authenticity. The extracted data is stored in structured CSV files via an MCP (Model Context Protocol) server.

## Key Features

- **Automated PDF Monitoring**: Watches directories for new documents and processes them automatically
- **Page-by-Page LLM Analysis**: Uses sophisticated prompts to extract genuine sensory experiences
- **Cultural Context Inference**: Automatically detects historical locale, era, register, and weather conditions
- **Quality Filtering**: Applies strict validation to ensure extracted terms are authentic sensory descriptions
- **Dual Processing Modes**: Standard extraction and culturally-aware extraction with period-appropriate vocabulary
- **Comprehensive Reporting**: Generates detailed markdown and JSON reports of processing results
- **File Management**: Organizes processed files with timestamped directories

## Architecture

```
docs/           → PDF files to process
   ↓ (monitoring)
sense_ingest_docs.py → PDF → Text → Context Inference → LLM → Validation
   ↓ (via MCP)
mcp_server.py   → CSV files in data/
   ↓ (success)
docs_done/      → Processed PDFs with timestamps
   ↓ (reports)
exports/ingest_reports/ → Processing reports
```

## Prerequisites

- **Python 3.12+**
- **Ollama** running locally (default: `http://localhost:11434`)
- **MCP Server** must be running before starting ingestion
- **spaCy English model** for Named Entity Recognition: `python -m spacy download en_core_web_sm`

## Installation

1. **Clone and setup:**
   ```bash
   git clone <repository>
   cd sense-bank-starter
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```

2. **Start Ollama and pull model:**
   ```bash
   ollama serve
   ollama pull llama3.1:8b  # or your preferred model
   ```

## Configuration

### Environment Variables

```bash
# LLM Configuration
export OPENAI_API_BASE="http://localhost:11434/v1"  # Ollama endpoint
export OPENAI_API_KEY="ollama"                      # Ollama key
export SENSEBANK_MODEL="llama3.1:8b"               # Model to use

# Data Directory
export SENSEBANK_DATA_DIR="./data"                 # CSV output directory
```

### Rules File (`ingest_rules.yaml`)

Configure context detection patterns:

```yaml
locales:
  Japan:
    - "\\bHeian|Kyoto|Nara|Yamato\\b"
  "Abbasid Baghdad":
    - "\\bBaghdad\\b"
  Venice:
    - "\\bVenice|Venezia|lagoon\\b"

eras:
  Heian:
    - "\\bHeian\\b"
    - "\\b(79[4-9]|8\\d\\d|11[0-7]\\d)\\b"
  Song:
    - "\\bSong\\b"
  Mughal:
    - "\\bMughal\\b"

registers:
  court: ["\\bcourt\\b", "\\bpalace\\b"]
  maritime: ["\\brope|tar|rigging|quay\\b"]
  monastic: ["\\bmonk|abbey|temple\\b"]
  # ... additional patterns
```

## Usage

### 1. Start the MCP Server (Required First)
```bash
python -u -m mcp_server.mcp_sense_bank
```

### 2. Start the Sense Ingestor
```bash
python -u -m sense_ingest_docs.sense_ingest_docs \
  --docs-dir docs \
  --default-locale "China" \
  --default-era "Song" \
  --move success --done-dir docs_done --fail-dir docs_fail --group-by-date \
  --report-dir exports/ingest_reports --report-format md --report-index \
  --log-level INFO --only-new
```

### 3. Add PDF Files
Simply drop PDF files into the `docs/` directory. The ingestor will:
- Detect new files automatically
- Process them page-by-page
- Extract sensory descriptions
- Move completed files to `docs_done/YYYYMMDD/`
- Generate reports in `exports/ingest_reports/`

## Command Line Options

### Core Options
- `--docs-dir DIR`: Directory to monitor for PDFs (default: `docs`)
- `--data-dir DIR`: Directory for CSV output (default: `data`)
- `--once`: Process once and exit (default: continuous monitoring)
- `--only-new`: Skip files already processed

### Context Defaults
- `--default-locale TEXT`: Default cultural locale
- `--default-era TEXT`: Default historical era  
- `--default-register TEXT`: Default social register (default: `common`)
- `--default-weather TEXT`: Default weather conditions (default: `any`)
- `--rules FILE`: YAML rules file for context detection

### File Management
- `--move {success,always,never}`: When to move processed files (default: `success`)
- `--done-dir DIR`: Success directory (default: `docs_done`)
- `--fail-dir DIR`: Failure directory (default: `docs_fail`)
- `--group-by-date`: Organize by date in subdirectories

### Processing Options
- `--no-ocr`: Disable OCR for image-based PDFs
- `--ocr-lang LANG`: OCR language (default: `eng`)
- `--bypass-mcp`: Write directly to CSV (bypasses MCP server)
- `--dry-run`: Process but don't save results

### Reporting
- `--report-dir DIR`: Report output directory
- `--report-format {md,json,both}`: Report format (default: `md`)
- `--report-index`: Generate index file

### Debugging
- `--log-level {DEBUG,INFO,WARNING,ERROR}`: Logging level
- `--log-file FILE`: Log to file
- `--heartbeat SECONDS`: Progress update interval (default: 15)

## Processing Pipeline

### 1. PDF Text Extraction
- Converts PDFs to text using `pdfplumber`
- Supports OCR for image-based content
- Processes page-by-page for optimal LLM handling

### 2. Context Inference
The system automatically detects:
- **Locale**: Geographic/cultural setting (Japan, Venice, Abbasid Baghdad, etc.)
- **Era**: Historical period (Heian, Song, Mughal, etc.)
- **Register**: Social context (court, maritime, monastic, market, etc.)
- **Weather**: Environmental conditions (rain, monsoon, winter snow, etc.)

### 3. LLM Processing
Each page is processed through sophisticated prompts that:
- Focus on genuine physical sensory experiences
- Filter out emotions, body parts, and abstract concepts
- Apply cultural context for period-appropriate vocabulary
- Extract 3-5 high-quality terms per page

### 4. Quality Validation
Extracted terms undergo rigorous filtering:
- **Category Requirements**: Must contain appropriate sensory vocabulary
- **Cultural Authenticity**: Validated against historical context
- **Length Filtering**: Minimum 6 characters for quality
- **Banned Terms**: Removes generic or inappropriate extractions

### 5. Data Storage
- Deduplicated across pages
- Stored via MCP server to structured CSV files
- Organized by locale in `data/` directory

## Sensory Categories

The system extracts five types of sensory experiences:

### Smell
- **Target**: Scents, odors, aromas, fragrances
- **Examples**: "acrid smoke", "sweet incense", "putrid decay", "aromatic oils"
- **Vocabulary**: fragrant, musty, smoky, pungent, aromatic

### Sound  
- **Target**: Audio qualities and characteristics
- **Examples**: "thunderous roar", "soft whisper", "distant chiming", "ceremonial bells"
- **Vocabulary**: piercing, melodic, muffled, resonant, thunderous

### Taste
- **Target**: Flavors experienced in the mouth
- **Examples**: "bitter medicine", "sweet honey", "metallic blood", "herbal remedies"
- **Vocabulary**: bitter, sweet, sour, salty, savory, metallic

### Touch
- **Target**: Physical sensations and textures
- **Examples**: "rough bark", "silky cloth", "burning heat", "smooth jade"
- **Vocabulary**: rough, smooth, soft, hard, hot, cold, warm, cool

### Sight
- **Target**: Visual qualities and luminosity
- **Examples**: "blazing fire", "dim shadows", "brilliant gold", "golden glow"
- **Vocabulary**: blazing, dim, brilliant, pale, vivid, gleaming, shimmering

## Cultural Context System

### Supported Historical Settings

#### Locales
- **Japan**: Heian court culture, seasonal aesthetics
- **China**: Song dynasty refinement, tea culture
- **Abbasid Baghdad**: Islamic golden age, scholarly pursuits
- **Venice**: Maritime culture, trade networks
- **Andes**: Andean civilizations, mountain environments

#### Eras
- **Heian (794-1185)**: Japanese court refinement
- **Song (960-1279)**: Chinese cultural sophistication  
- **Mughal (1526-1857)**: Indo-Islamic imperial culture
- **Chimú (900-1470)**: Andean coastal civilization

#### Registers
- **Court**: Palace life, ceremonial contexts, luxury goods
- **Maritime**: Ships, ports, ocean environments
- **Monastic**: Religious institutions, scholarly pursuits
- **Market**: Commercial spaces, everyday trade
- **Festival**: Celebrations, public gatherings
- **Military**: Warfare, encampments, weapons
- **Garden**: Cultivated nature, aesthetic spaces

## Output Format

### CSV Structure
```csv
term,category,locale,era,register,weather,notes
"Aromatic Incense",smell,"Japan","Heian",court,any,"Temple ceremony context"
"Thunderous Bells",sound,"Japan","Heian",monastic,rain,"Morning prayers"
```

### Report Structure
- **Processing Summary**: Files, success rates, timing
- **Category Breakdown**: Distribution across sensory types
- **Page Analysis**: Detailed per-page extraction results
- **Context Detection**: Inferred cultural settings
- **Quality Metrics**: Validation and filtering statistics

## Monitoring and Logs

### Real-time Monitoring
```
[13:47:34] INFO Page 21/21: 1 items (LLM 2.52s)
[13:47:34] INFO Deduped to 12 unique items (smell:4, sound:4, touch:2, sight:2)
[13:47:34] INFO Writing 12 items via MCP…
[13:47:34] INFO ✓ Ingest finished in 59.56s (added=12, exists=0, errors=0, total=12)
[13:47:34] INFO Moved The Wanya Hu.pdf → docs_done/20250819
[13:47:34] INFO Report: exports/ingest_reports/The Wanya Hu/20250819_134734.md
```

### Log Levels
- **DEBUG**: Detailed extraction and validation info
- **INFO**: Processing progress and summaries
- **WARNING**: Non-fatal issues and fallbacks
- **ERROR**: Processing failures and exceptions

## Natural Language Interface

Use `strands_chat.py` to query the processed data:

```bash
python -u strands_chat.py
```

This provides a conversational interface to explore the extracted sensory database.

## Troubleshooting

### Common Issues

**"MCP server not running"**
- Start the MCP server first: `python -u -m mcp_server.mcp_sense_bank`

**"No terms extracted"**
- Check LLM connectivity: verify Ollama is running
- Review log level: use `--log-level DEBUG` for detailed info
- Validate input: ensure PDFs contain extractable text

**"Context inference failing"**
- Update rules file: add patterns for your specific documents
- Set explicit defaults: use `--default-locale` and `--default-era`

**"Quality filtering too strict"**
- Review ban terms in `sense_ingest_docs.py`
- Check category requirements for your content type

### Performance Optimization

- **Heartbeat**: Adjust `--heartbeat` for progress updates
- **Batch Size**: Process fewer files simultaneously
- **OCR**: Disable with `--no-ocr` if not needed
- **Model**: Use smaller models for faster processing

## Development

### Module Structure
```
sense_ingest_docs/
├── sense_ingest_docs.py    # Main ingestion engine
├── context_inference.py    # Cultural context detection
├── llm_interface.py        # LLM prompts and API calls
├── reporting.py           # Report generation
├── csv_data_manager.py    # Data management
├── pdf_watcher.py         # File monitoring
└── mcp_client.py         # MCP communication
```

### Adding New Cultural Contexts

1. **Update rules file**: Add regex patterns for new locales/eras
2. **Extend vocabulary**: Add context-specific terms in `llm_interface.py`
3. **Test extraction**: Validate with sample documents
4. **Update documentation**: Document new cultural contexts

## License

Apache Source License V2

## Contributing

Send me a note at  if you'd like to contribute!