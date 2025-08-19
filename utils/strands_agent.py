#!/usr/bin/env python3
import os, json
from dotenv import load_dotenv
load_dotenv()

from strands import Agent
from strands.models.openai import OpenAIModel
from strands.tools.mcp.mcp_client import MCPClient
from mcp.client.streamable_http import streamablehttp_client

# ---------- helper: robust print for MCP results (works across SDK versions)
def print_tool_result(res, label="RESULT"):
    # If agent returns a dict/list/string directly
    if isinstance(res, (dict, list)):
        print(f"\n{label}:", json.dumps(res, ensure_ascii=False, indent=2)); return
    if isinstance(res, str):
        try:
            obj = json.loads(res)
            print(f"\n{label}:", json.dumps(obj, ensure_ascii=False, indent=2))
        except Exception:
            print(f"\n{label}:", res)
        return

    # Newer SDKs: structured content
    sc = getattr(res, "structuredContent", None) or getattr(res, "structured_content", None)
    if sc:
        print(f"\n{label} (structured):", json.dumps(sc, ensure_ascii=False, indent=2)); return

    # Fallback: regular content list
    contents = getattr(res, "content", None)
    if contents:
        last = contents[-1]
        text = getattr(last, "text", None)
        if isinstance(text, str):
            try:
                obj = json.loads(text)
                print(f"\n{label} (parsed JSON):", json.dumps(obj, ensure_ascii=False, indent=2)); return
            except Exception:
                pass
        parts = [getattr(c, "text", str(c)) for c in contents if getattr(c, "text", None)]
        print(f"\n{label} (text):", "\n".join(parts) if parts else str(res)); return

    print(f"\n{label} (raw):", str(res))

# ---------- model (OpenAI-compatible; Ollama via client_args)
model = OpenAIModel(
    model_id=os.getenv("SENSEBANK_MODEL", "llama3.1:8b"),
    client_args={
        "api_key": os.getenv("OPENAI_API_KEY", "ollama"),
        "base_url": os.getenv("OPENAI_API_BASE", "http://localhost:11434/v1"),
    },
    params={"temperature": 0.7},
)

# ---------- MCP client (matches server at http://127.0.0.1:8000/mcp/)
MCP_URL = os.getenv("MCP_URL", "http://127.0.0.1:8000/mcp/")
mcp_client = MCPClient(lambda: streamablehttp_client(MCP_URL))

with mcp_client:
    # Fetch tool descriptors from the MCP server
    tools = list(mcp_client.list_tools_sync())

    # Build agent with those tools
    agent = Agent(model=model, tools=tools)

    # (A) Direct call via MCP client
    res = mcp_client.call_tool_sync(
        tool_use_id="sense-001",
        name="sense_suggest",
        arguments={"locale":"Japan","era":"Heian","weather":"rain","register":"court","n":6,"memory_key":"CH8"},
    )
    print_tool_result(res, "SUGGEST")

    # (B) Call via Agent’s tool proxy
    out = agent.tool.sense_rewrite(
        locale="Japan", era="Heian", weather="rain", register="court", n=6, memory_key="CH8",
        text="After the storm, the palace courtyard smelled nice and everything felt calm. She crossed the veranda and tried to breathe it in."
    )
    print_tool_result(out, "REWRITE")

    # (C) Let the agent choose tools from a natural prompt
    reply = agent("Rewrite the passage in Heian court register after rain; use sensory cues; avoid anachronisms. "
                  "Passage: 'After the storm, the palace courtyard smelled nice…'")
    print("\nAGENT:", reply)
