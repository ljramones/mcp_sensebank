import argparse, json, os, sys
from sense_bank.tools.sensory import (  # <-- use sensory_tool
    load_corpus, suggest_sensory, rewrite_with_sensory, load_memory, save_memory
)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
MEM_DIR  = os.path.join(os.path.dirname(os.path.dirname(__file__)), "memory")

def cmd_suggest(args):
    if getattr(args, "model", None):
        os.environ["SENSEBANK_MODEL"] = args.model

    rows = load_corpus(DATA_DIR)
    memory_key = args.memory_key
    memory = load_memory(MEM_DIR, memory_key) if memory_key else {}

    items = suggest_sensory(
        rows=rows,                     # <-- rows, not df
        locale=args.locale,
        era=args.era,
        weather=args.weather,
        register=args.register,
        n=args.n,
        memory=memory
    )
    print(json.dumps(items, ensure_ascii=False, indent=2))

    if memory_key:
        used = memory.get("used_terms", [])
        used.extend([x["term"] for x in items])
        memory["used_terms"] = sorted(set(used))
        save_memory(MEM_DIR, memory_key, memory)

def cmd_rewrite(args):
    if getattr(args, "model", None):
        os.environ["SENSEBANK_MODEL"] = args.model

    rows = load_corpus(DATA_DIR)
    memory_key = args.memory_key
    memory = load_memory(MEM_DIR, memory_key) if memory_key else {}

    suggestions = suggest_sensory(
        rows=rows,                     # <-- rows, not df
        locale=args.locale,
        era=args.era,
        weather=args.weather,
        register=args.register,
        n=args.n,
        memory=memory
    )

    text = args.text
    if not text and args.text_file:
        with open(args.text_file, "r", encoding="utf-8") as f:
            text = f.read()
    if not text:
        print("Provide --text or --text-file", file=sys.stderr)
        sys.exit(2)

    out = rewrite_with_sensory(text, suggestions)
    result = {
        "suggestions": suggestions,
        "rewrite": out.get("rewrite"),
        "model": out.get("model"),
        "reasoning": out.get("reasoning")
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))

    if memory_key:
        used = memory.get("used_terms", [])
        used.extend([x["term"] for x in suggestions])
        memory["used_terms"] = sorted(set(used))
        save_memory(MEM_DIR, memory_key, memory)

def main():
    p = argparse.ArgumentParser(prog="sense-bank")
    sub = p.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--locale", required=True, help="e.g., Japan, Andes")
    common.add_argument("--era", required=True, help="e.g., Heian, Chimú")
    common.add_argument("--weather", default="any", help="e.g., rain, dry, any")
    common.add_argument("--register", default="common", help="court, common, ritual, military, etc.")
    common.add_argument("--n", type=int, default=6, help="number of suggestions")
    common.add_argument("--memory-key", default=None, help="persist/avoid repeats across runs")

    common.add_argument("--model", default=None, help="override SENSEBANK_MODEL for this run")

    s1 = sub.add_parser("suggest", parents=[common])
    s1.set_defaults(func=cmd_suggest)

    s2 = sub.add_parser("rewrite", parents=[common])
    s2.add_argument("--text", default=None, help="inline text snippet")
    s2.add_argument("--text-file", default=None, help="path to a snippet file")
    s2.set_defaults(func=cmd_rewrite)

    args = p.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
