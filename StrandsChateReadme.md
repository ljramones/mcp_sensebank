# Strands Chat - Sense Bank Natural Language Interface

A powerful command-line interface for interacting with the Sense Bank database through natural language processing and structured commands. This tool provides comprehensive access to query, analyze, and manipulate the sensory data collected by the Sense Ingestor.

## Overview

Strands Chat serves as the primary interface for exploring your sensory database. It combines the power of Large Language Models with direct database operations to provide both conversational interactions and precise data management capabilities.

## Features

- **Natural Language Processing**: Rewrite text using contextual sensory cues
- **Advanced Search**: Semantic and structured search across the sensory database
- **Data Management**: Add, edit, delete, and move sensory records
- **Export Capabilities**: Generate reports in CSV, Markdown, and Excel formats
- **Data Validation**: Lint and clean CSV data with comprehensive checks
- **Statistical Analysis**: Generate corpus summaries and insights
- **Context-Aware Operations**: Maintain cultural and temporal context across sessions

## Prerequisites

- **Python 3.12+**
- **MCP Server** running (`mcp_sense_bank.py`)
- **Ollama** with your preferred model (default: `llama3.1:8b`)
- **Strands framework** and dependencies from `requirements.txt`

## Installation

1. **Ensure dependencies are installed:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up environment variables:**
   ```bash
   export SENSEBANK_MODEL="llama3.1:8b"           # LLM model to use
   export OPENAI_API_BASE="http://localhost:11434/v1"  # Ollama endpoint
   export OPENAI_API_KEY="ollama"                 # Ollama API key
   export MCP_URL="http://127.0.0.1:8000/mcp/"   # MCP server endpoint
   ```

3. **Start the MCP server (required):**
   ```bash
   python -u -m mcp_server.mcp_sense_bank
   ```

4. **Launch Strands Chat:**
   ```bash
   python -u strands_chat.py
   ```

## Usage

### Starting a Session

When you launch Strands Chat, you'll see:
```
Sense Bank chat ready. Commands:
...
> 
```

The system maintains a context throughout your session with default settings:
```json
{
  "locale": "Japan",
  "era": "Heian", 
  "weather": "rain",
  "register": "court",
  "n": 6,
  "memory_key": "SESSION"
}
```

### Command Categories

## Context Management

### `/context`
Display current session context settings.

### `/set key=val [...]`
Update context variables for the session.

**Examples:**
```bash
/set locale=China era=Song register=market n=10
/set no_repeat=true exclude=smell,sound
```

**Available keys:**
- `locale`: Cultural/geographic setting
- `era`: Historical period
- `register`: Social context 
- `weather`: Environmental conditions
- `n`: Number of suggestions to return
- `no_repeat`: Avoid repeating recent suggestions
- `exclude`: Categories to exclude from suggestions
- `exclude_terms`: Specific terms to avoid

## Content Generation

### `/suggest [key=val ...]`
Generate contextual sensory cues based on current or specified context.

**Examples:**
```bash
/suggest                                    # Use current context
/suggest locale=Venice era=Medieval n=8     # Override context
/suggest exclude=smell,sound no_repeat=true # Exclude categories
/suggest exclude_terms="bright light,loud noise"
```

### `/rewrite <text>`
Enhance text with contextual sensory descriptions.

**Examples:**
```bash
/rewrite The garden was beautiful in the morning.
# Plain text (no slash) also works as rewrite:
The temple bells rang across the courtyard.
```

**Multi-line input (use backslash to continue):**
```bash
> /rewrite The ceremony began at dawn. \
… Monks gathered in the temple hall. \
… Incense filled the air.
```

## Data Exploration

### `/search q="..." [options]`
Semantic search across the sensory database.

**Parameters:**
- `q`: Search query (required)
- `locale`: Filter by locale
- `era`: Filter by era
- `category`: Filter by sensory category
- `top_k`: Number of results (default: 10)

**Examples:**
```bash
/search q="temple bells" top_k=5
/search q="silk texture" locale=China era=Song
/search q="rain sounds" category=sound
```

### `/list [key=val ...]`
List and filter sensory records with structured criteria.

**Parameters:**
- `category`: Sensory category (smell, sound, taste, touch, sight)
- `locale`: Cultural/geographic setting
- `era`: Historical period
- `weather`: Weather conditions
- `register`: Social context
- `term_contains`: Term substring search
- `notes_contains`: Notes substring search
- `file`: Source file filter
- `limit`: Maximum results (default: 50)
- `sort`: Sort fields (e.g., "category,term")

**Examples:**
```bash
/list category=smell locale=Japan limit=20
/list era=Heian register=court sort="category,term"
/list term_contains="silk" notes_contains="ceremony"
/list file="ancient_texts.pdf"
```

### `/stats [key=val ...]`
Generate statistical summaries of the sensory corpus.

**Parameters:**
- Filtering: Same as `/list` (category, locale, era, etc.)
- `group_by`: Grouping fields (default: "locale,era,category")
- `sort_by`: Sort by count, unique_terms, or group
- `desc`: Descending sort (true/false)
- `top`: Limit to top N groups (default: 10)
- `examples`: Include example terms (true/false)
- `examples_per`: Examples per group (default: 3)

**Examples:**
```bash
/stats                                      # Overall corpus summary
/stats group_by=category sort_by=count desc=true
/stats locale=Japan group_by=era,register examples=true
/stats category=smell top=5 examples_per=5
```

## Data Management

### `/add key=val [...]`
Add new sensory records to the database.

**Required parameters:**
- `term`: Sensory description
- `category`: Sensory category
- `locale`: Cultural setting
- `era`: Historical period

**Optional parameters:**
- `weather`: Weather conditions
- `register`: Social context
- `notes`: Additional context
- `file`: Source file reference

**Examples:**
```bash
/add term="Aromatic Incense" category=smell locale=Japan era=Heian register=temple
/add term="Silk Rustling" category=sound locale=China era=Song weather=calm notes="Court ceremony"
```

### `/edit key=val [...]`
Edit existing sensory records.

**Required parameters (for matching):**
- `match_term`: Term to find
- `match_category`: Category to find  
- `match_locale`: Locale to find
- `match_era`: Era to find

**Fields to update:**
- `term`, `category`, `locale`, `era`, `weather`, `register`, `notes`, `file`
- `dry_run`: Preview changes without applying

**Examples:**
```bash
/edit match_term="Old Term" match_category=smell match_locale=Japan match_era=Heian term="New Term"
/edit match_term="Silk Sound" match_category=sound match_locale=China match_era=Song notes="Updated context" dry_run=true
```

### `/delete key=val [...]`
Delete sensory records from the database.

**Required parameters:**
- `match_term`: Term to find
- `match_category`: Category to find
- `match_locale`: Locale to find  
- `match_era`: Era to find

**Optional parameters:**
- `file`: Additional file filter
- `dry_run`: Preview deletion without executing

**Examples:**
```bash
/delete match_term="Unwanted Term" match_category=smell match_locale=Japan match_era=Heian
/delete match_term="Test Entry" match_category=sound match_locale=China match_era=Song dry_run=true
```

### `/move key=val [...]`
Move records between files or update locale information.

**Required parameters:**
- `match_term`: Term to find
- `match_category`: Category to find
- `match_locale`: Locale to find
- `match_era`: Era to find

**Optional parameters:**
- `dest_locale`: New locale
- `dest_file`: Destination file
- `update_locale_field`: Update the locale field (true/false)
- `dry_run`: Preview move without executing

**Examples:**
```bash
/move match_term="Silk Sound" match_category=sound match_locale=Japan match_era=Heian dest_locale=China
/move match_term="Temple Bell" match_category=sound match_locale=Japan match_era=Heian dest_file=china_sensory.csv dry_run=true
```

## Data Export

### `/csv key=val [...]`
Export filtered data to CSV format.

**Parameters:**
- Filtering: Same as `/list`
- `out`: Output filename (required)
- `append`: Append to existing file (true/false)
- `include_header`: Include CSV header (true/false)

**Examples:**
```bash
/csv locale=Japan era=Heian out=heian_court.csv
/csv category=smell out=all_smells.csv include_header=true
/csv locale=China append=true out=china_data.csv
```

### `/md key=val [...]`
Export filtered data to Markdown table format.

**Parameters:**
- Filtering: Same as `/list`
- `out`: Output filename (required)
- `append`: Append to existing file (true/false)
- `include_header`: Include table header (true/false)

**Examples:**
```bash
/md locale=Venice era=Medieval out=venice_report.md
/md category=sound,smell out=audio_olfactory.md include_header=true
```

### `/xlsx key=val [...]`
Export filtered data to Excel format.

**Parameters:**
- Filtering: Same as `/list`
- `out`: Output filename (required)
- `sheet_name`: Worksheet name (default: "SenseBank")
- `overwrite`: Overwrite existing file (true/false)

**Examples:**
```bash
/xlsx locale=Japan out=japan_sensory.xlsx sheet_name="Heian Period"
/xlsx category=smell,taste out=flavor_aroma.xlsx overwrite=true
```

### `/pack key=val [...]`
Export both CSV and Markdown for a complete dataset.

**Parameters:**
- Filtering: Same as `/list`
- `out`: Base filename (auto-generated if not provided)

**Examples:**
```bash
/pack locale=Japan era=Heian register=court
# Creates: pack_japan_heian_court.csv and pack_japan_heian_court.md

/pack locale=China era=Song out=song_dynasty
# Creates: song_dynasty.csv and song_dynasty.md
```

## Data Maintenance

### `/validate [key=val ...]`
Validate and clean CSV data with comprehensive checks.

**Parameters:**
- `file`: Specific file to validate
- `fix_whitespace`: Clean whitespace issues (true/false)
- `normalize_case`: Standardize capitalization (true/false)
- `dedupe`: Remove duplicates within file (true/false)
- `global_dedupe`: Remove duplicates across all files (true/false)
- `allowed_categories`: Comma-separated valid categories
- `allowed_registers`: Comma-separated valid registers
- `allowed_weather`: Comma-separated valid weather conditions
- `dry_run`: Preview fixes without applying (default: false)
- `limit`: Maximum rows to process (default: 100)

**Examples:**
```bash
/validate                                   # Validate all files
/validate file=japan_sensory.csv dry_run=true
/validate fix_whitespace=true normalize_case=true dedupe=true
/validate allowed_categories="smell,sound,taste,touch,sight" limit=500
/validate allowed_weather="rain,snow,clear,cloudy" global_dedupe=true
```

### `/importmd path=... [options]`
Import sensory data from Markdown files.

**Parameters:**
- `path`: Path to Markdown file (required)
- `file`: Target CSV file
- `on_duplicate`: How to handle duplicates (skip/update/error)
- `dry_run`: Preview import without executing

**Examples:**
```bash
/importmd path=external_data.md file=imported_sensory.csv
/importmd path=research_notes.md on_duplicate=update dry_run=true
```

## Advanced Usage

### Context-Aware Sessions

Maintain consistent cultural context throughout your session:

```bash
# Set up Song Dynasty China context
/set locale=China era=Song register=scholar weather=autumn

# Generate suggestions in this context
/suggest n=10

# Search within this context
/search q="tea ceremony" 

# Rewrite text with this context
/rewrite The scholar prepared for the examination.
```

### Complex Data Analysis

Combine commands for comprehensive analysis:

```bash
# Analyze Japanese sensory data
/stats locale=Japan group_by=era,register examples=true
/list locale=Japan category=smell limit=20 sort="era,term"
/md locale=Japan out=japan_analysis.md

# Compare different periods
/stats era=Heian group_by=category
/stats era=Song group_by=category  
```

### Batch Operations

Use validation and editing for data cleanup:

```bash
# Comprehensive data validation
/validate fix_whitespace=true normalize_case=true dedupe=true dry_run=true
/validate global_dedupe=true allowed_categories="smell,sound,taste,touch,sight"

# Move misclassified entries
/move match_term="Silk Texture" match_category=sight match_locale=Japan match_era=Heian category=touch
```

## Command Reference

### Input Conventions

- **Multi-line input**: Use backslash (`\`) at end of line to continue
- **Quoted parameters**: Use quotes for values with spaces or commas
- **Boolean values**: `true/false`, `yes/no`, `1/0`
- **Lists**: Comma-separated values in quotes: `exclude="smell,sound"`

### Common Parameters

| Parameter | Description | Example Values |
|-----------|-------------|----------------|
| `locale` | Cultural/geographic setting | `Japan`, `China`, `Venice`, `Abbasid Baghdad` |
| `era` | Historical period | `Heian`, `Song`, `Mughal`, `Medieval` |
| `register` | Social context | `court`, `market`, `temple`, `maritime` |
| `category` | Sensory type | `smell`, `sound`, `taste`, `touch`, `sight` |
| `weather` | Environmental conditions | `rain`, `snow`, `clear`, `monsoon` |

### Error Handling

The system provides detailed error messages and suggestions:

```bash
> /add term="Test" category=smell
Missing: locale, era

> /search
Parse error: missing required parameter 'q'

> /validate allowed_weather="bad,quotes
Parse error: unterminated quoted string
Tip: put comma-lists in quotes, e.g. allowed_weather="dry season,winter snow"
```

## Integration with Sense Ingestor

Strands Chat works seamlessly with the Sense Ingestor pipeline:

1. **Sense Ingestor** processes PDFs and extracts sensory data
2. **MCP Server** stores data in structured CSV files  
3. **Strands Chat** provides interface for exploration and analysis

### Typical Workflow

```bash
# After ingestion, explore new data
/stats group_by=file sort_by=count desc=true top=5

# Examine recent additions
/list file="latest_document.pdf" limit=20

# Generate contextual content
/set locale=Japan era=Heian 
/suggest n=8
/rewrite The temple ceremony began at dawn.

# Export findings
/pack locale=Japan era=Heian out=heian_temple_sensory
```

## Tips and Best Practices

### Efficient Searching
- Use specific terms in search queries for better results
- Combine semantic search (`/search`) with structured filtering (`/list`)
- Use `top_k` parameter to control result quantity

### Data Management
- Always use `dry_run=true` for destructive operations first
- Validate data regularly with `/validate`
- Use consistent terminology across locale and era fields

### Context Management
- Set session context early for consistent suggestions
- Use `/context` to check current settings
- Adjust `n` parameter based on your needs (3-10 typically optimal)

### Export Strategies
- Use `/pack` for complete datasets
- Use `/md` for readable reports
- Use `/xlsx` for analysis in spreadsheet tools
- Use `/csv` for data processing pipelines

## Troubleshooting

### Common Issues

**"MCP connection failed"**
- Ensure MCP server is running: `python -u -m mcp_server.mcp_sense_bank`
- Check MCP_URL environment variable

**"No results found"**
- Verify data exists with `/stats`
- Check spelling of locale/era names
- Use `/list` without filters to see all data

**"LLM not responding"**
- Verify Ollama is running: `ollama serve`
- Check OPENAI_API_BASE and SENSEBANK_MODEL settings
- Try simpler queries first

**"Parse error in command"**
- Use quotes around values with spaces or commas
- Check command syntax in help text
- Ensure all required parameters are provided

### Performance Tips

- Use `limit` parameter for large datasets
- Filter by specific locales/eras for faster queries
- Use `dry_run` to test operations before execution

## Exit

Use `/quit`, `/exit`, or `Ctrl+C` to exit the interface.

---

*Strands Chat provides a powerful interface for exploring and manipulating your sensory database. Combine structured commands with natural language processing to unlock insights from your historical text corpus.*