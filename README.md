# Sense Bank (local-first writing aid)

A tiny agent-style tool that suggests **period-appropriate sensory details** (smell, sound, tactile)
for your scenes and optionally performs "show, don't tell" rewrites using a local LLM via an
**OpenAI-compatible** endpoint (Ollama or a LiteLLM proxy).

## Quick start

```bash
cd sense-bank-starter
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Configure model (local-first)

**Option A: Ollama OpenAI endpoint (fully local)**
```bash
# In another terminal
ollama serve  # or `brew services start ollama`

# Default config expects this:
export OPENAI_API_BASE="http://localhost:11434/v1"
export OPENAI_API_KEY="ollama"     # any string works for Ollama
export SENSEBANK_MODEL="llama3.1"  # or "llama3.1:8b" etc.
```

**Option B: LiteLLM proxy (mix local + paid models)**
```bash
export OPENAI_API_BASE="http://localhost:4000/v1"
export OPENAI_API_KEY="sk-..."     # LiteLLM key if required
export SENSEBANK_MODEL="local/llama3.1"  # or anthropic/..., google/...
```

## Run

### 1) Get suggestions only
```bash
python -m sense_bank.agent suggest --locale Japan --era Heian --weather rain --register court --n 6
```

### 2) Rewrite a snippet using suggestions
```bash
python -m sense_bank.agent rewrite --locale Japan --era Heian --weather rain --register court --text "The courtyard smelled nice after the storm."
```

### 3) Use a memory key to avoid repeats across sessions/chapters
```bash
python -m sense_bank.agent suggest --locale Japan --era Heian --weather rain --register court --n 6 --memory-key CH8
python -m sense_bank.agent rewrite --locale Japan --era Heian --weather rain --register court --text "$(cat snippet.txt)" --memory-key CH8
```

Outputs print to stdout; suggestions also save to `memory/<key>.json` when you use `--memory-key`.

## Files
- `sense_bank/agent.py` – CLI entrypoint
- `sense_bank/tools/sensory.py` – data loader, filtering, memory, and LLM-assisted rewrite
- `data/jp_sensory.csv` – seed data for Heian Japan
- `data/andes_sensory.csv` – seed data for Andes/Chimú
- `memory/` – persisted choices to reduce repetition

## Notes
- CSVs are small seeds. Add to them freely; categories can be `smell|sound|tactile|visual|taste` (you're free to extend).
- The rewrite step is optional; if no model env is configured, the tool will skip LLM calls and just print suggestions.
