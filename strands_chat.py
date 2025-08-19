#!/usr/bin/env python3
import os, json, shlex
from dotenv import load_dotenv
import re

from strands import Agent
from strands.models.openai import OpenAIModel
from strands.tools.mcp.mcp_client import MCPClient
from mcp.client.streamable_http import streamablehttp_client

load_dotenv()


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", (s or "all").strip().lower()).strip("_")


def print_tool_result(res, label="RESULT"):
    if isinstance(res, (dict, list)): print(f"\n{label}:", json.dumps(res, ensure_ascii=False, indent=2)); return
    if isinstance(res, str):
        try:
            print(f"\n{label}:", json.dumps(json.loads(res), ensure_ascii=False, indent=2))
        except Exception:
            print(f"\n{label}:", res);
            return
        return
    sc = getattr(res, "structuredContent", None) or getattr(res, "structured_content", None)
    if sc: print(f"\n{label} (structured):", json.dumps(sc, ensure_ascii=False, indent=2)); return
    contents = getattr(res, "content", None)
    if contents:
        text = getattr(contents[-1], "text", None)
        if isinstance(text, str):
            try:
                print(f"\n{label} (parsed JSON):", json.dumps(json.loads(text), ensure_ascii=False, indent=2));
                return
            except Exception:
                pass
        parts = [getattr(c, "text", str(c)) for c in contents if getattr(c, "text", None)]
        print(f"\n{label} (text):", "\n".join(parts) if parts else str(res))
        return
    print(f"\n{label} (raw):", str(res))


MODEL_ID = os.getenv("SENSEBANK_MODEL", "llama3.1:8b")
BASE_URL = os.getenv("OPENAI_API_BASE", "http://localhost:11434/v1")
API_KEY = os.getenv("OPENAI_API_KEY", "ollama")
MCP_URL = os.getenv("MCP_URL", "http://127.0.0.1:8000/mcp/")

model = OpenAIModel(model_id=MODEL_ID, client_args={"api_key": API_KEY, "base_url": BASE_URL},
                    params={"temperature": 0.7})
mcp_client = MCPClient(lambda: streamablehttp_client(MCP_URL))

context = {"locale": "Japan", "era": "Heian", "weather": "rain", "register": "court", "n": 6, "memory_key": "SESSION"}


def help_text():
    return (
        "Commands:\n"
        "  /context                         show current context\n"
        "  /suggest [key=val ...]            list cues; supports exclude=smell,sound exclude_terms=\"a,b\" no_repeat=true\n"
        "  /set key=val [...]                now also accepts no_repeat=true/false and exclude=...\n"
        "  /pack key=val [...]               export both CSV+MD for a locale/era/register; same keys as /list (plus out=base)\n"
        "  /rewrite <text>                  rewrite given text using cues\n"
        "  /importmd path=... [file=... on_duplicate=skip|update|error dry_run=true]\n"
        "  /search q=\"...\" [locale=..., era=..., category=..., top_k=10]\n"
        "  /add key=val [...]               add a new row (term,category,locale,era required; optional weather,register,notes,file)\n"
        "  /edit key=val [...]              edit a row: match_term,match_category,match_locale,match_era REQUIRED\n"
        "                                   fields to change: term,category,locale,era,weather,register,notes,file,[dry_run=true]\n"
        "  /delete key=val [...]            delete matching row: match_term,match_category,match_locale,match_era REQUIRED\n"
        "                                   optional: file=..., dry_run=true\n"
        "  /list key=val [...]              filter rows; supports category/locale/era/weather/register,\n"
        "                                   term_contains=..., notes_contains=..., file=..., limit=50, sort=\"category,term\"\n"
        "  /md  key=val [...]               export to Markdown table; keys like /list plus out=..., append=true, include_header=false\n"
        "  /xlsx key=val [...]              export to .xlsx; keys like /list plus out=..., sheet_name=SenseBank, overwrite=false\n"
        "  /csv key=val [...]                export filtered rows to CSV; keys like /list plus out=..., append=true, include_header=false\n"
        "  /validate [key=val ...]          lint CSVs; keys: file=..., fix_whitespace=true, normalize_case=true, dedupe=true,\n"
        "                                   global_dedupe=true, allowed_categories=..., allowed_registers=..., allowed_weather=..., dry_run=false, limit=100\n"
        "  /move key=val [...]              move a row: match_term,match_category,match_locale,match_era REQUIRED;\n"
        "                                   optional: dest_locale=..., dest_file=..., update_locale_field=true, dry_run=true\n"
        "  /stats [key=val ...]              corpus summary; keys like /list plus group_by=locale,era,category,\n"
        "                                    sort_by=count|unique_terms|group, desc=true, top=10, examples=true, examples_per=3\n"
        "  /quit                            exit\n"
        "Plain text (no slash) is treated as /rewrite <text>.\n"
    )


with mcp_client:
    tools = list(mcp_client.list_tools_sync())
    agent = Agent(model=model, tools=tools)

    print("Sense Bank chat ready. " + help_text())
    while True:
        try:
            buf = ""
            while True:
                raw = input("> " if not buf else "… ").rstrip()
                if raw.endswith("\\"):
                    buf += raw[:-1] + " "  # drop the trailing backslash, add a space
                    continue
                line = (buf + raw).strip()
                break
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            break
        if not line: continue
        if line in ("/quit", "/exit"): break

        if line == "/context":
            print(json.dumps(context, ensure_ascii=False, indent=2))
            continue

        if line.startswith("/search"):
            args = {}
            for tok in shlex.split(line[len("/search"):].strip()):
                if "=" in tok:
                    k,v = tok.split("=",1); args[k]=v
            if "top_k" in args:
                try: args["top_k"] = int(args["top_k"])
                except: args["top_k"] = 10
            res = mcp_client.call_tool_sync("sense-ctx", "sense_search", args)
            print_tool_result(res, "SEARCH")
            continue

        if line.startswith("/set"):
            parts = line.split()[1:]
            for p in parts:
                if "=" in p:
                    k, v = p.split("=", 1)
                    if k == "n":
                        try:
                            context[k] = int(v)
                        except:
                            context[k] = 6
                    elif k in ("no_repeat",):
                        context[k] = v.lower() in ("1", "true", "yes", "y")
                    else:
                        context[k] = v
            print("OK")
            continue

        if line.startswith("/importmd"):
            args = {}
            for tok in shlex.split(line[len("/importmd"):].strip()):
                if "=" in tok:
                    k, v = tok.split("=", 1);
                    args[k] = v
            if "dry_run" in args:
                args["dry_run"] = args["dry_run"].lower() in ("1", "true", "yes", "y")
            res = mcp_client.call_tool_sync("sense-ctx", "sense_import_md", args)
            print_tool_result(res, "IMPORTMD")
            continue

        if line.startswith("/pack "):
            args = {}
            for tok in shlex.split(line[len("/pack "):]):
                if "=" in tok:
                    k, v = tok.split("=", 1)
                    args[k] = v
            # Build default basename if not provided
            base = args.get("out")
            if not base:
                base = "pack_" + "_".join(filter(None, [
                    _slug(args.get("locale", context.get("locale", ""))),
                    _slug(args.get("era", context.get("era", ""))),
                    _slug(args.get("register", "")),
                ]))
            # CSV
            csv_args = dict(args)
            csv_args["out"] = base + ".csv"
            md_args = dict(args)
            md_args["out"] = base + ".md"
            r1 = mcp_client.call_tool_sync("sense-ctx", "sense_export", csv_args)
            r2 = mcp_client.call_tool_sync("sense-ctx", "sense_export_md", md_args)
            print_tool_result(r1, "PACK-CSV")
            print_tool_result(r2, "PACK-MD")
            continue

        if line.startswith("/suggest"):
            # start from context; allow overrides
            args = dict(context)
            rest = line[len("/suggest"):].strip()
            if rest:
                for tok in shlex.split(rest):
                    if "=" in tok:
                        k, v = tok.split("=", 1)
                        if k == "n":
                            try:
                                args[k] = int(v)
                            except:
                                args[k] = args.get("n", 6)
                        elif k in ("no_repeat",):
                            args[k] = v.lower() in ("1", "true", "yes", "y")
                        else:
                            args[k] = v
            res = mcp_client.call_tool_sync("sense-ctx", "sense_suggest", args)
            print_tool_result(res, "SUGGEST")
            continue

        if line.startswith("/add "):
            args = {}
            for tok in shlex.split(line[len("/add "):]):
                if "=" in tok:
                    k, v = tok.split("=", 1)
                    args[k] = v
            required = ("term", "category", "locale", "era")
            missing = [k for k in required if k not in args]
            if missing: print("Missing:", ", ".join(missing)); continue
            res = mcp_client.call_tool_sync("sense-ctx", "sense_add", args)
            print_tool_result(res, "ADDED")
            continue

        if line.startswith("/edit "):
            args = {}
            for tok in shlex.split(line[len("/edit "):]):
                if "=" in tok:
                    k, v = tok.split("=", 1)
                    args[k] = v
            required = ("match_term", "match_category", "match_locale", "match_era")
            missing = [k for k in required if k not in args]
            if missing: print("Missing:", ", ".join(missing)); continue
            if "dry_run" in args:
                args["dry_run"] = args["dry_run"].lower() in ("1", "true", "yes", "y")
            res = mcp_client.call_tool_sync("sense-ctx", "sense_edit", args)
            print_tool_result(res, "EDIT")
            continue

        if line.startswith("/delete "):
            args = {}
            for tok in shlex.split(line[len("/delete "):]):
                if "=" in tok:
                    k, v = tok.split("=", 1)
                    args[k] = v
            required = ("match_term", "match_category", "match_locale", "match_era")
            missing = [k for k in required if k not in args]
            if missing: print("Missing:", ", ".join(missing)); continue
            if "dry_run" in args:
                args["dry_run"] = args["dry_run"].lower() in ("1", "true", "yes", "y")
            res = mcp_client.call_tool_sync("sense-ctx", "sense_delete", args)
            print_tool_result(res, "DELETE")
            continue

        if line.startswith("/csv "):
            args = {}
            for tok in shlex.split(line[len("/csv "):]):
                if "=" in tok:
                    k, v = tok.split("=", 1)
                    args[k] = v
            # normalize types
            if "limit" in args:
                try:
                    args["limit"] = int(args["limit"])
                except:
                    args["limit"] = 0
            if "append" in args:
                args["append"] = args["append"].lower() in ("1", "true", "yes", "y")
            if "include_header" in args:
                args["include_header"] = args["include_header"].lower() in ("1", "true", "yes", "y")

            res = mcp_client.call_tool_sync("sense-ctx", "sense_export", args)
            print_tool_result(res, "CSV")
            continue

        if line.startswith("/md "):
            args = {}
            for tok in shlex.split(line[len("/md "):]):
                if "=" in tok:
                    k, v = tok.split("=", 1)
                    args[k] = v
            if "limit" in args:
                try:
                    args["limit"] = int(args["limit"])
                except:
                    args["limit"] = 0
            if "append" in args:
                args["append"] = args["append"].lower() in ("1", "true", "yes", "y")
            if "include_header" in args:
                args["include_header"] = args["include_header"].lower() in ("1", "true", "yes", "y")
            res = mcp_client.call_tool_sync("sense-ctx", "sense_export_md", args)
            print_tool_result(res, "MD")
            continue

        if line.startswith("/validate"):
            args = {}
            try:
                for tok in shlex.split(line[len("/validate"):].strip()):
                    if "=" in tok:
                        k, v = tok.split("=", 1)
                        args[k] = v
            except ValueError as e:
                print("Parse error:", e)
                print('Tip: put comma-lists in quotes, e.g. allowed_weather="dry season,winter snow"')
                continue
            # booleans
            for b in ("fix_whitespace", "normalize_case", "dedupe", "global_dedupe", "dry_run"):
                if b in args: args[b] = args[b].lower() in ("1", "true", "yes", "y")
            # ints
            if "limit" in args:
                try:
                    args["limit"] = int(args["limit"])
                except:
                    args["limit"] = 100
            res = mcp_client.call_tool_sync("sense-ctx", "sense_validate", args)
            print_tool_result(res, "VALIDATE")
            continue

        if line.startswith("/move "):
            args = {}
            for tok in shlex.split(line[len("/move "):]):
                if "=" in tok:
                    k, v = tok.split("=", 1)
                    args[k] = v
            required = ("match_term", "match_category", "match_locale", "match_era")
            missing = [k for k in required if k not in args]
            if missing:
                print("Missing:", ", ".join(missing))
                continue
            if "update_locale_field" in args:
                args["update_locale_field"] = args["update_locale_field"].lower() in ("1", "true", "yes", "y")
            if "dry_run" in args:
                args["dry_run"] = args["dry_run"].lower() in ("1", "true", "yes", "y")
            res = mcp_client.call_tool_sync("sense-ctx", "sense_move", args)
            print_tool_result(res, "MOVE")
            continue

        if line.startswith("/xlsx "):
            args = {}
            for tok in shlex.split(line[len("/xlsx "):]):
                if "=" in tok:
                    k, v = tok.split("=", 1)
                    args[k] = v
            if "limit" in args:
                try:
                    args["limit"] = int(args["limit"])
                except:
                    args["limit"] = 0
            if "overwrite" in args:
                args["overwrite"] = args["overwrite"].lower() in ("1", "true", "yes", "y")
            if "include_header" in args:
                args["include_header"] = args["include_header"].lower() in ("1", "true", "yes", "y")
            res = mcp_client.call_tool_sync("sense-ctx", "sense_export_sheet", args)
            print_tool_result(res, "XLSX")
            continue

        if line.startswith("/stats"):
            args = {}
            for tok in shlex.split(line[len("/stats"):].strip()):
                if "=" in tok:
                    k, v = tok.split("=", 1)
                    args[k] = v
            # booleans / ints
            if "desc" in args: args["desc"] = args["desc"].lower() in ("1", "true", "yes", "y")
            if "examples" in args: args["examples"] = args["examples"].lower() in ("1", "true", "yes", "y")
            if "top" in args:
                try:
                    args["top"] = int(args["top"])
                except:
                    args["top"] = 0
            if "examples_per" in args:
                try:
                    args["examples_per"] = int(args["examples_per"])
                except:
                    args["examples_per"] = 3
            res = mcp_client.call_tool_sync("sense-ctx", "sense_stats", args)
            print_tool_result(res, "STATS")
            continue

        if line.startswith("/list "):
            args = {}
            for tok in shlex.split(line[len("/list "):]):
                if "=" in tok:
                    k, v = tok.split("=", 1)
                    args[k] = v
            if "limit" in args:
                try:
                    args["limit"] = int(args["limit"])
                except:
                    args["limit"] = 50
            res = mcp_client.call_tool_sync("sense-ctx", "sense_list", args)
            print_tool_result(res, "LIST")
            continue

        if line.startswith("/rewrite "):
            text = line[len("/rewrite "):]
        else:
            text = line

        call = dict(context)
        call["text"] = text
        out = agent.tool.sense_rewrite(**call)
        print_tool_result(out, "REWRITE")
