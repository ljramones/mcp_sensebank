# MCP Sense Bank Server

A Model Context Protocol (MCP) server that provides comprehensive data management and AI-powered tools for the Sense Bank sensory database. This server acts as the central hub for storing, querying, and analyzing culturally-contextualized sensory descriptions extracted from historical documents.

## Overview

The MCP Sense Bank Server implements the Model Context Protocol to provide a standardized interface for managing sensory data. It serves as the backend for both the Sense Ingestor (which feeds data in) and Strands Chat (which provides user interaction). The server maintains CSV-based storage organized by cultural locale and provides sophisticated tools for data manipulation, export, and AI-powered content generation.

## Features

### Data Management
- **CRUD Operations**: Add, edit, delete, and move sensory records
- **Intelligent Routing**: Automatically organizes data by cultural locale
- **Deduplication**: Prevents duplicate entries across the database
- **Validation**: Comprehensive data quality checks and cleanup tools

### Search & Discovery
- **Semantic Search**: Fuzzy matching across terms and notes
- **Structured Filtering**: Multi-dimensional filtering by category, locale, era, etc.
- **Statistical Analysis**: Corpus-wide analytics and insights

### AI-Powered Content Generation
- **Contextual Suggestions**: Generate culturally-appropriate sensory terms
- **Text Enhancement**: Rewrite text with period-authentic sensory descriptions
- **Memory Management**: Track usage to avoid repetition

### Export Capabilities
- **Multiple Formats**: CSV, Markdown, and Excel export
- **Flexible Filtering**: Export subsets based on any criteria
- **Batch Operations**: Pack complete datasets for analysis

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Sense Ingestor │───▶│  MCP Server      │◀───│  Strands Chat   │
│                 │    │                  │    │                 │
│ • PDF Processing│    │ • Data Storage   │    │ • User Interface│
│ • LLM Extraction│    │ • AI Integration │    │ • Query Tools   │
│ • Context Detect│    │ • Export Tools   │    │ • Content Gen   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │     Data Store   │
                       │                  │
                       │ • CSV Files      │
                       │ • Memory Files   │
                       │ • Export Files   │
                       └──────────────────┘
```

## Prerequisites

- **Python 3.12+**
- **FastMCP framework** (included in requirements.txt)
- **Sense Bank core tools** (in `sense_bank/` directory)
- **Optional**: `xlsxwriter` for Excel export support

## Installation

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   # Optional for Excel export:
   pip install xlsxwriter
   ```

2. **Set up directory structure:**
   ```bash
   mkdir -p data memory exports
   ```

3. **Configure environment:**
   ```bash
   # Optional: Set custom data directory
   export SENSEBANK_DATA_DIR="./data"
   ```

## Usage

### Starting the Server

```bash
# Start the MCP server
python -u -m mcp_server.mcp_sense_bank

# Or using direct path
python -u mcp_server/mcp_sense_bank.py
```

The server will start on `http://127.0.0.1:8000/mcp/` by default.

### Server Endpoints

The server provides MCP-compliant endpoints:
- `POST /mcp/tools/list` - List available tools
- `POST /mcp/tools/call` - Execute tool operations
- `GET /mcp/health` - Health check

## Tools Reference

The server provides 16 comprehensive tools for data management and AI operations:

### Data Management Tools

#### `sense_add`
Add new sensory records to the database.

**Required Parameters:**
- `term`: Sensory description
- `category`: One of: smell, sound, taste, touch, sight
- `locale`: Cultural/geographic setting
- `era`: Historical period

**Optional Parameters:**
- `weather`: Environmental conditions (default: "any")
- `register`: Social context (default: "common")
- `notes`: Additional context
- `file`: Specific CSV file (auto-routed if not provided)

**Example Response:**
```json
{
  "status": "added",
  "file": "data/jp_sensory.csv",
  "record": {
    "term": "Aromatic Incense",
    "category": "smell",
    "locale": "Japan",
    "era": "Heian",
    "weather": "any",
    "register": "temple",
    "notes": "Buddhist ceremony"
  },
  "created_file": false
}
```

#### `sense_edit`
Modify existing sensory records.

**Required Parameters (for matching):**
- `match_term`: Current term value
- `match_category`: Current category
- `match_locale`: Current locale  
- `match_era`: Current era

**Update Parameters:**
- `term`, `category`, `locale`, `era`, `weather`, `register`, `notes`, `file`
- `dry_run`: Preview changes (default: false)

#### `sense_delete`
Remove sensory records from the database.

**Required Parameters:**
- `match_term`, `match_category`, `match_locale`, `match_era`

**Optional Parameters:**
- `file`: Restrict to specific file
- `dry_run`: Preview deletion

#### `sense_move`
Transfer records between files or update locale information.

**Required Parameters:**
- `match_term`, `match_category`, `match_locale`, `match_era`

**Optional Parameters:**
- `dest_locale`: New locale (updates file routing)
- `dest_file`: Explicit destination file
- `update_locale_field`: Update the locale field (default: true)
- `dry_run`: Preview operation

### Search & Discovery Tools

#### `sense_search`
Semantic search across the sensory database.

**Parameters:**
- `q`: Search query (required)
- `locale`, `era`, `category`: Filter parameters
- `file`: Restrict to specific file
- `top_k`: Number of results (default: 10)
- `weight_term`: Term matching weight (default: 0.7)
- `weight_notes`: Notes matching weight (default: 0.3)

**Example Response:**
```json
{
  "status": "ok",
  "query": "temple bells",
  "top_k": 5,
  "results": [
    {
      "score": 0.8945,
      "row": {
        "term": "Resonant Bells",
        "category": "sound",
        "locale": "Japan",
        "era": "Heian"
      }
    }
  ]
}
```

#### `sense_list`
Structured filtering and listing of records.

**Filter Parameters:**
- `category`, `locale`, `era`, `weather`, `register`: Exact matches
- `term_contains`, `notes_contains`: Substring searches
- `file`: File restriction
- `limit`: Maximum results (default: 50)
- `sort`: Sort fields, e.g., "category,term"

#### `sense_stats`
Generate statistical summaries of the corpus.

**Parameters:**
- All filtering parameters from `sense_list`
- `group_by`: Grouping fields (default: "locale,era,category")
- `sort_by`: Sort by count, unique_terms, or group
- `desc`: Descending sort (default: true)
- `top`: Limit to top N groups
- `examples`: Include example terms (default: true)
- `examples_per`: Examples per group (default: 3)

### AI-Powered Tools

#### `sense_suggest`
Generate contextually-appropriate sensory suggestions.

**Parameters:**
- `locale`, `era`: Cultural context (required)
- `weather`: Environmental conditions (default: "any")
- `register`: Social context (default: "common")
- `n`: Number of suggestions (default: 6)
- `memory_key`: Session memory for tracking usage
- `exclude`: Categories to exclude (comma-separated)
- `exclude_terms`: Specific terms to avoid
- `no_repeat`: Avoid recently used terms

**Example Response:**
```json
[
  {
    "term": "Aromatic Incense",
    "category": "smell",
    "locale": "Japan",
    "era": "Heian",
    "notes": "Temple ceremony context"
  },
  {
    "term": "Soft Chanting",
    "category": "sound",
    "locale": "Japan", 
    "era": "Heian",
    "notes": "Monastic prayers"
  }
]
```

#### `sense_rewrite`
Enhance text with contextual sensory descriptions.

**Parameters:**
- `locale`, `era`, `text`: Required context and input
- `weather`, `register`, `n`: Optional parameters
- `memory_key`: Session tracking
- `exclude`, `exclude_terms`, `no_repeat`: Filtering options

**Example Response:**
```json
{
  "suggestions": [...],
  "rewrite": "The temple ceremony began at dawn, with aromatic incense drifting through the halls and soft chanting echoing from the prayer rooms.",
  "model": "llama3.1:8b",
  "reasoning": "Enhanced with period-appropriate sensory details"
}
```

### Export Tools

#### `sense_export`
Export filtered data to CSV format.

**Parameters:**
- All filtering parameters from `sense_list`
- `out`: Output filename (relative to exports/)
- `append`: Append to existing file (default: false)
- `include_header`: Include CSV header (default: true)

#### `sense_export_md`
Export filtered data to Markdown table format.

**Parameters:**
- Same as `sense_export` but for Markdown output
- Automatically escapes pipe characters in content

#### `sense_export_sheet`
Export filtered data to Excel format.

**Parameters:**
- Same filtering as other export tools
- `sheet_name`: Worksheet name (default: "SenseBank")
- `overwrite`: Overwrite existing file (default: false)

**Note:** Requires `xlsxwriter` package: `pip install xlsxwriter`

### Maintenance Tools

#### `sense_validate`
Comprehensive data validation and cleanup.

**Parameters:**
- `file`: Specific file to validate (default: all files)
- `fix_whitespace`: Clean whitespace issues (default: false)
- `normalize_case`: Standardize capitalization (default: false)
- `dedupe`: Remove duplicates within files (default: false)
- `global_dedupe`: Report cross-file duplicates (default: false)
- `allowed_categories`: Valid categories (comma-separated)
- `allowed_registers`: Valid registers (comma-separated)
- `allowed_weather`: Valid weather conditions (comma-separated)
- `dry_run`: Preview fixes (default: true)
- `limit`: Maximum problems to report (default: 100)

**Built-in Defaults:**
- **Registers**: court, common, ritual, market, monastic, maritime, festival, military, scholar, garden
- **Weather**: any, rain, summer, winter, monsoon, dry season, winter snow, dawn, night

#### `sense_import_md`
Import sensory data from Markdown table files.

**Parameters:**
- `path`: Path to Markdown file (required)
- `file`: Target CSV file (optional, auto-routed by locale)
- `on_duplicate`: skip, update, or error (default: "skip")
- `dry_run`: Preview import (default: true)

## Data Organization

### File Structure
```
data/
├── jp_sensory.csv          # Japan locale data
├── china_sensory.csv       # China locale data
├── andes_sensory.csv       # Andes locale data
├── venice_sensory.csv      # Venice locale data
└── ...                     # Additional locales

memory/
├── SESSION.json           # User session memory
└── ...                    # Other memory keys

exports/
├── reports/               # Generated reports
├── sense_export_*.csv     # Exported CSV files
├── sense_export_*.md      # Exported Markdown
└── sense_export_*.xlsx    # Exported Excel files
```

### CSV Schema
All CSV files follow the standardized header:
```csv
term,category,locale,era,weather,register,notes
"Aromatic Incense",smell,"Japan","Heian",any,temple,"Buddhist ceremony"
"Resonant Bells",sound,"Japan","Heian",rain,monastic,"Morning prayers"
```

### Auto-Routing Logic
The server automatically routes records to appropriate files based on locale:
- `Japan` → `jp_sensory.csv`
- `Andes` → `andes_sensory.csv`
- Other locales → `{locale_name}_sensory.csv`

## Memory Management

The server maintains session memory for AI operations:
- **Usage Tracking**: Remembers recently suggested terms
- **Repetition Avoidance**: Prevents suggesting the same terms repeatedly
- **Session Persistence**: Maintains context across multiple requests

Memory files are stored in JSON format in the `memory/` directory.

## Error Handling

The server provides detailed error responses:

```json
{
  "status": "error",
  "error": "Missing required fields: term, category",
  "file": "data/jp_sensory.csv"
}
```

Common error types:
- `not_found`: Record doesn't exist
- `duplicate`: Duplicate entry detected
- `missing_required`: Required fields missing
- `file_not_found`: Specified file doesn't exist
- `validation_error`: Data validation failed

## Integration

### With Sense Ingestor
The Sense Ingestor calls `sense_add` to store extracted sensory data:

```python
# Ingestor adds extracted terms
for term_data in extracted_terms:
    result = mcp_client.call_tool("sense-ctx", "sense_add", term_data)
```

### With Strands Chat
Strands Chat uses all server tools for comprehensive data interaction:

```python
# Search for terms
results = mcp_client.call_tool("sense-ctx", "sense_search", {
    "q": "temple bells",
    "locale": "Japan"
})

# Generate suggestions
suggestions = mcp_client.call_tool("sense-ctx", "sense_suggest", {
    "locale": "Japan",
    "era": "Heian",
    "n": 5
})
```

## Performance Considerations

### Memory Usage
- CSV files are loaded entirely into memory for operations
- Large datasets (>10MB) may require memory optimization
- Consider splitting large locale files by era or category

### Concurrent Access
- File operations use atomic writes via temporary files
- Multiple clients can read simultaneously
- Write operations are serialized per file

### Optimization Tips
- Use `limit` parameters for large result sets
- Cache frequently accessed files in client applications
- Use specific file parameters to avoid scanning all CSVs

## API Examples

### Adding a Sensory Record
```bash
curl -X POST http://127.0.0.1:8000/mcp/tools/call \
  -H "Content-Type: application/json" \
  -d '{
    "name": "sense_add",
    "arguments": {
      "term": "Aromatic Incense",
      "category": "smell",
      "locale": "Japan",
      "era": "Heian",
      "register": "temple",
      "notes": "Buddhist ceremony"
    }
  }'
```

### Searching the Database
```bash
curl -X POST http://127.0.0.1:8000/mcp/tools/call \
  -H "Content-Type: application/json" \
  -d '{
    "name": "sense_search",
    "arguments": {
      "q": "temple bells",
      "locale": "Japan",
      "top_k": 5
    }
  }'
```

### Generating Suggestions
```bash
curl -X POST http://127.0.0.1:8000/mcp/tools/call \
  -H "Content-Type: application/json" \
  -d '{
    "name": "sense_suggest",
    "arguments": {
      "locale": "Japan",
      "era": "Heian",
      "register": "court",
      "n": 6
    }
  }'
```

## Monitoring and Logs

The server provides logging for:
- Tool execution timing
- File operations
- Error conditions
- Memory management

Enable debug logging for detailed operation tracking:
```bash
PYTHONPATH=. python -u mcp_server/mcp_sense_bank.py --log-level DEBUG
```

## Troubleshooting

### Common Issues

**"File not found" errors**
- Ensure the `data/` directory exists
- Check file permissions
- Verify CSV file names match locale routing

**"Duplicate entry" warnings**
- Use `sense_validate` with `dedupe=true` to clean data
- Check for case sensitivity issues
- Verify matching criteria in edit/delete operations

**"Memory errors with large files**
- Split large CSV files by era or category
- Increase system memory allocation
- Use `limit` parameters in queries

**"Excel export fails"**
- Install xlsxwriter: `pip install xlsxwriter`
- Check file permissions in exports directory
- Verify disk space availability

### Performance Issues

**Slow search operations**
- Use more specific filter parameters
- Reduce `top_k` values
- Consider indexing for very large datasets

**High memory usage**
- Monitor CSV file sizes
- Use file-specific operations when possible
- Implement pagination for large result sets

## Development

### Adding New Tools
1. Define the tool function with `@mcp.tool()` decorator
2. Add comprehensive parameter validation
3. Implement proper error handling
4. Update this documentation

### Testing
```bash
# Test individual tools
python -c "
from mcp_sense_bank import sense_add
result = sense_add('Test Term', 'smell', 'Japan', 'Heian')
print(result)
"

# Test server endpoints
curl http://127.0.0.1:8000/mcp/health
```

### Data Migration
For major schema changes:
1. Use `sense_export` to backup all data
2. Implement migration scripts
3. Use `sense_import_md` or custom tools to restore data
4. Validate with `sense_validate`

---

The MCP Sense Bank Server provides the backbone for sophisticated sensory data management, combining traditional database operations with AI-powered content generation to support historical text analysis and cultural research.