#!/usr/bin/env python3
import csv
import sys, hashlib, re
from pathlib import Path
from typing import  Mapping
import yaml
import shutil
from datetime import datetime
from collections import Counter
from contextlib import contextmanager

from strands.tools.mcp.mcp_client import MCPClient
from mcp.client.streamable_http import streamablehttp_client

import os, json, time, uuid, logging, atexit, requests
from typing import Any, Optional, ContextManager, cast
from contextlib import ExitStack
import threading
from urllib.parse import urlparse, parse_qsl, urlencode, urlunparse
from typing import Dict, List, Tuple


# ------------------ config / env ------------------
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "http://localhost:11434/v1")
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY", "ollama")
MODEL_ID        = os.getenv("SENSEBANK_MODEL", "llama3.1:8b")

STATE_PATH      = Path(".sense_ingest_state.json")  # remembers processed files

DEFAULT_REGISTER = "common"
DEFAULT_WEATHER  = "any"

ALLOWED_CATS = {"smell","sound","taste","touch","sight"}
BAN_TERMS = {"fear","sorrow","disgust","horrific","light","darkness","smile","eyes","face","location","sound","sight","taste"}  # tune
MIN_TERM_CHARS = 4

SENSORY_HINTS = {
  "smell": {"scent","odor","aroma","acrid","resin","smoke","incense","camphor","must","rank"},
  "sound": {"clang","ring","murmur","whinny","shriek","rumble","drone","hiss","chime","thud"},
  "taste": {"bitter","sweet","coppery","salty","sour","umami","astringent"},
  "touch": {"warmth","chill","clammy","grit","greasy","oily","rough","silken"},
  "sight": {"glare","glint","sheen","milky","ashen","verdigris","sooty","saffron"},
}

# Where the Sense-Bank CSVs live (override via flag or env)
DATA_DIR = Path(os.getenv("SENSEBANK_DATA_DIR", Path(__file__).parent / "data"))

# CSV schema
HEADER = ["term", "category", "locale", "era", "weather", "register", "notes"]

# Optional mapping for nicer filenames per locale
DEFAULT_FILES = {
    "Japan": "jp_sensory.csv",
    "Andes": "andes_sensory.csv",
    "China": "cn_sensory.csv",
    "Venice": "venice_sensory.csv",
    "Abbasid Baghdad": "baghdad_sensory.csv",
    # add more as you like…
}

# ---- config / globals ----
MCP_URL = os.getenv("MCP_URL", "http://127.0.0.1:8000/mcp").rstrip("/")

_MCP_CLIENT = None            # type: MCPClient | None
_MCP_LOCK = threading.Lock()
_MCP_STACK = ExitStack()      # manages context lifetime
MCP_SESSION_HEADER = os.getenv("MCP_SESSION_HEADER")  # optional override, e.g. "OpenAI-Session-Id"
_MCP_SES: Optional[requests.Session] = None   # keep a single session only
_MCP_DEBUG_ONCE = True
# Add/ensure these near your globals:
_MCP_SESSION_HEADER_NAMES = tuple(x for x in (
    os.getenv("MCP_SESSION_HEADER"),      # optional override
    "OpenAI-Transport-Id", "X-OpenAI-Transport-Id",
    "OpenAI-Session-Id",  "X-OpenAI-Session-Id",
    "X-MCP-Session", "X-Session",
) if x)
_MCP_COOKIE_KEYS = (
    "mcp-session", "mcp_session", "sessionid", "session",
    "transport", "transport_id", "openai_transport_id"
)

# Try several SSE endpoints; overrideable via env
_MCP_SSE_PATHS = tuple(
    p for p in (
        os.getenv("MCP_SSE_PATH"),
        "", "/", "/events", "/stream", "/sse"
    ) if p is not None
)

_MCP_TRANSPORT_HEADER = os.getenv("MCP_TRANSPORT_HEADER", "OpenAI-Transport-Id")
_MCP_SESSION_PARAM_NAMES = ("sessionId", "transportId")  # common names servers check


_MCP_SESSION: Optional[requests.Session] = None
_MCP_SESSION_ID: Optional[str] = None
_MCP_INITIALIZED: bool = False

# ------------------ MCP call ------------------

def _poll_for_tool_result_debug(session: requests.Session, timeout: float) -> Dict[str, Any]:
    """Debug version - logs everything the server returns."""
    logger = logging.getLogger("sense-ingest")
    deadline = time.time() + timeout
    poll_count = 0

    logger.info(f"Starting polling with {timeout}s timeout")

    while time.time() < deadline:
        poll_count += 1
        remaining = deadline - time.time()

        try:
            logger.info(f"Poll {poll_count}: {remaining:.1f}s remaining")
            response = session.get(MCP_URL, timeout=5)

            logger.info(f"Poll {poll_count}: HTTP {response.status_code}")
            logger.info(f"Poll {poll_count}: Content-Type: {response.headers.get('content-type')}")
            logger.info(f"Poll {poll_count}: Content-Length: {len(response.text)}")
            logger.info(f"Poll {poll_count}: Response sample: {response.text[:200]}")

            if response.status_code == 200:
                # Try to parse as JSON
                try:
                    data = response.json()
                    logger.info(
                        f"Poll {poll_count}: Parsed JSON structure: {list(data.keys()) if isinstance(data, dict) else type(data)}")

                    result = _extract_tool_result(data)
                    if result:
                        logger.info(f"Poll {poll_count}: Extracted result: {result}")
                        if result.get("status") not in ("pending", None):
                            logger.info(f"SUCCESS after {poll_count} polls")
                            return result
                        else:
                            logger.info(f"Poll {poll_count}: Result still pending")
                    else:
                        logger.info(f"Poll {poll_count}: No result extracted from JSON")

                except json.JSONDecodeError as e:
                    logger.info(f"Poll {poll_count}: JSON decode failed: {e}")

                    # Try SSE format
                    if 'data:' in response.text:
                        logger.info(f"Poll {poll_count}: Trying SSE parsing")
                        result = _parse_sse_for_tool_result(response.text)
                        if result:
                            logger.info(f"Poll {poll_count}: SSE result: {result}")
                            if result.get("status") not in ("pending", None):
                                return result
                    else:
                        logger.info(f"Poll {poll_count}: No 'data:' found in response")

        except requests.RequestException as e:
            logger.error(f"Poll {poll_count} failed: {e}")

        # Wait before next poll
        time.sleep(0.5)  # Slower polling for debugging

    logger.error(f"TIMEOUT after {poll_count} polls in {timeout}s")
    return {"status": "error", "error": f"timeout after {poll_count} polls in {timeout}s"}


def get_mcp_session() -> requests.Session:
    """Get or create a persistent MCP session with initialization."""
    global _MCP_SESSION, _MCP_SESSION_ID, _MCP_INITIALIZED

    if _MCP_SESSION is None:
        logger = logging.getLogger("sense-ingest")

        # Create persistent session
        _MCP_SESSION = requests.Session()
        _MCP_SESSION.headers.update({
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
            "Connection": "keep-alive",
        })

        # Get session ID
        try:
            response = _MCP_SESSION.get(MCP_URL, timeout=10)
            if 'mcp-session-id' in response.headers:
                _MCP_SESSION_ID = response.headers['mcp-session-id']
                _MCP_SESSION.headers['mcp-session-id'] = _MCP_SESSION_ID
                logger.info(f"Got session ID: {_MCP_SESSION_ID}")
        except Exception as e:
            logger.error(f"Failed to get session ID: {e}")

        # Initialize the MCP session
        if not _MCP_INITIALIZED:
            _initialize_mcp_session(_MCP_SESSION)

    return _MCP_SESSION


def _initialize_mcp_session(session: requests.Session) -> bool:
    """Initialize MCP session with capabilities exchange."""
    global _MCP_INITIALIZED, _MCP_SESSION_ID
    logger = logging.getLogger("sense-ingest")

    try:
        # Step 1: Initialize request
        init_request = {
            "jsonrpc": "2.0",
            "id": f"init-{uuid.uuid4().hex}",
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {}  # We want to use tools
                },
                "clientInfo": {
                    "name": "sense-ingest",
                    "version": "1.0.0"
                }
            }
        }

        # Add session ID if we have one
        if _MCP_SESSION_ID:
            init_request["params"]["sessionId"] = _MCP_SESSION_ID

        logger.info("Sending MCP initialize request")
        response = session.post(MCP_URL, json=init_request, timeout=15)

        if response.status_code == 200:
            logger.info("MCP initialization successful")

            # Step 2: Send initialized notification
            notification = {
                "jsonrpc": "2.0",
                "method": "notifications/initialized",
                "params": {}
            }

            if _MCP_SESSION_ID:
                notification["params"]["sessionId"] = _MCP_SESSION_ID

            session.post(MCP_URL, json=notification, timeout=10)
            logger.info("Sent initialized notification")

            _MCP_INITIALIZED = True
            return True
        else:
            logger.error(f"MCP initialization failed: {response.status_code} - {response.text}")
            return False

    except Exception as e:
        logger.error(f"MCP initialization error: {e}")
        return False



def _id_from_any_response(resp: requests.Response) -> Optional[str]:
    # 1) headers: accept any header whose name mentions session/transport/mcp
    for k, v in resp.headers.items():
        kl = k.lower()
        if ("session" in kl) or ("transport" in kl) or ("mcp" in kl):
            if v and isinstance(v, str) and len(v) >= 16:
                return v.strip()

    # 2) cookies: common cookie keys
    for ck in _MCP_COOKIE_KEYS:
        try:
            vv = resp.cookies.get(ck)  # type: ignore[arg-type]
            if vv:
                return str(vv).strip()
        except Exception:
            pass

    # 3) json bodies that might carry session info
    try:
        j = resp.json()
        if isinstance(j, dict):
            res = j.get("result") or {}
            for key in ("sessionId", "id"):
                v = res.get(key) or j.get(key)
                if v:
                    return str(v).strip()
    except Exception:
        pass
    return None

def mcp_call(tool_name: str, arguments: Dict[str, Any], timeout: float = 10.0) -> Dict[str, Any]:
    """
    Call MCP tool with immediate SSE response handling.
    """
    logger = logging.getLogger("sense-ingest")

    # Get initialized session
    session = get_mcp_session()

    if not _MCP_INITIALIZED:
        return {"status": "error", "error": "MCP session not initialized"}

    # Prepare tool call
    rpc_request = {
        "jsonrpc": "2.0",
        "id": f"tool-{uuid.uuid4().hex}",
        "method": "tools/call",
        "params": {
            "name": tool_name,
            "arguments": arguments,
        }
    }

    # Add session ID
    global _MCP_SESSION_ID
    if _MCP_SESSION_ID:
        rpc_request["params"]["sessionId"] = _MCP_SESSION_ID

    try:
        # Make the tool call
        response = session.post(MCP_URL, json=rpc_request, timeout=timeout)

        if response.status_code == 200:
            content_type = response.headers.get('content-type', '')

            # Handle SSE response (which contains immediate result)
            if 'text/event-stream' in content_type:
                return _parse_immediate_sse_response(response.text)

            # Handle regular JSON response
            else:
                try:
                    data = response.json()
                    return _extract_tool_result(data) or {"status": "ok", "data": data}
                except json.JSONDecodeError:
                    return {"status": "error", "error": "Invalid JSON response"}

        else:
            return {
                "status": "error",
                "error": f"HTTP {response.status_code}",
                "body": response.text[:200]
            }

    except requests.RequestException as e:
        return {"status": "error", "error": f"Request failed: {str(e)}"}


def _parse_immediate_sse_response(sse_text: str) -> Dict[str, Any]:
    """Parse SSE response that contains immediate results."""
    lines = sse_text.strip().split('\n')

    for line in lines:
        line = line.strip()
        if line.startswith('data: '):
            data_content = line[6:]  # Remove 'data: ' prefix

            try:
                # Parse the JSON-RPC response
                rpc_response = json.loads(data_content)

                # Extract the result
                if "result" in rpc_response:
                    result = rpc_response["result"]

                    # Method 1: Look for structuredContent first (most reliable)
                    if "structuredContent" in result:
                        structured = result["structuredContent"]
                        if isinstance(structured, dict) and "result" in structured:
                            struct_result = structured["result"]
                            if struct_result.get("type") == "json" and "json" in struct_result:
                                return struct_result["json"]

                    # Method 2: Look for content array with text containing JSON
                    if "content" in result and isinstance(result["content"], list):
                        for content_item in result["content"]:
                            if isinstance(content_item, dict) and content_item.get("type") == "text":
                                text_content = content_item.get("text", "")
                                try:
                                    # Parse the nested JSON
                                    inner_data = json.loads(text_content)
                                    if isinstance(inner_data, dict) and "json" in inner_data:
                                        return inner_data["json"]
                                    elif isinstance(inner_data, dict):
                                        return inner_data
                                except json.JSONDecodeError:
                                    continue

                    # Method 3: Return result directly if it looks valid
                    if isinstance(result, dict) and ("status" in result or "content" in result):
                        return result

                # Handle JSON-RPC error
                elif "error" in rpc_response:
                    error = rpc_response["error"]
                    return {
                        "status": "error",
                        "error": error.get("message", str(error)) if isinstance(error, dict) else str(error)
                    }

            except json.JSONDecodeError:
                continue

    # Couldn't parse anything useful
    return {"status": "error", "error": "Could not parse SSE response"}

def _extract_tool_result(data: Any) -> Optional[Dict[str, Any]]:
    """Extract tool result from MCP response."""
    if isinstance(data, dict):
        # Direct JSON-RPC result
        if "result" in data:
            result = data["result"]
            if isinstance(result, dict):
                # Check for tool call result
                if "content" in result:
                    # Parse the content
                    content = result["content"]
                    if isinstance(content, list) and content:
                        for item in content:
                            if isinstance(item, dict):
                                if "text" in item:
                                    try:
                                        # Try to parse text content as JSON
                                        text_data = json.loads(item["text"])
                                        return text_data
                                    except:
                                        pass
                                elif item.get("type") == "json" and "json" in item:
                                    return item["json"]

                # Direct result
                return result

        # Error response
        if "error" in data:
            return {"status": "error", "error": str(data["error"])}

    elif isinstance(data, list):
        # Try each item
        for item in data:
            result = _extract_tool_result(item)
            if result and result.get("status") != "error":
                return result

    return None


def _parse_sse_for_tool_result(sse_text: str) -> Optional[Dict[str, Any]]:
    """Parse SSE response for tool results."""
    lines = sse_text.strip().split('\n')

    for line in lines:
        line = line.strip()
        if line.startswith('data: '):
            data_content = line[6:]
            if data_content and data_content not in ['[DONE]', '']:
                try:
                    data = json.loads(data_content)
                    result = _extract_tool_result(data)
                    if result:
                        return result
                except json.JSONDecodeError:
                    continue

    return {"status": "pending"}


def _establish_session_properly(session: requests.Session) -> Optional[str]:
    """Establish session by getting the server-generated session ID."""
    global _MCP_SESSION_ID
    logger = logging.getLogger("sense-ingest")

    try:
        # Make a GET request to establish session
        response = session.get(MCP_URL, timeout=10)
        logger.debug(f"Session establishment GET: {response.status_code}")

        # Look for session ID in response headers (try multiple header names)
        session_id = None
        for header_name in ["X-Session-ID", "X-Transport-ID", "OpenAI-Session-Id",
                            "OpenAI-Transport-Id", "Session-ID", "Transport-ID"]:
            if header_name in response.headers:
                session_id = response.headers[header_name]
                logger.info(f"Found server session ID in {header_name}: {session_id}")
                break

        # Also check cookies
        if not session_id:
            for cookie in response.cookies:
                if any(name in cookie.name.lower() for name in ["session", "transport"]):
                    session_id = cookie.value
                    logger.info(f"Found server session ID in cookie {cookie.name}: {session_id}")
                    break

        # If we got a session ID from server, use it
        if session_id:
            _MCP_SESSION_ID = session_id
            # Add it to all future requests
            session.headers.update({
                "X-Session-ID": session_id,
                "X-Transport-ID": session_id,
                "OpenAI-Session-Id": session_id,
                "OpenAI-Transport-Id": session_id,
            })
            return session_id
        else:
            logger.warning("Server did not provide session ID, will try to use server-created one")

    except Exception as e:
        logger.error(f"Failed to establish session: {e}")

    return None

def _apply_session(s: requests.Session, sid: str) -> None:
    """Broadcast the server-issued id via headers and cookies for subsequent calls."""
    global _MCP_SESSION_ID
    _MCP_SESSION_ID = sid
    for h in _MCP_SESSION_HEADER_NAMES:
        s.headers[h] = sid
    # Also mirror into cookies for servers that bind via cookies on GET/poll
    try:
        for ck in _MCP_COOKIE_KEYS:
            s.cookies.set(ck, sid)
    except Exception:
        pass


def _close_mcp_client():
    """Close all managed contexts (including the MCP client) safely."""
    global _MCP_CLIENT
    with _MCP_LOCK:
        try:
            _MCP_STACK.close()
        except Exception as e:
            logging.getLogger("sense-ingest").debug("MCP close error: %s", e)
        finally:
            _MCP_CLIENT = None

atexit.register(_close_mcp_client)



def _default_file_for_locale(locale: str) -> str:
    key = (locale or "misc").strip()
    return DEFAULT_FILES.get(key, f"{key.lower().replace(' ', '_')}_sensory.csv")


def _csv_load_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        return [{k: (v or "") for k, v in row.items()} for row in r]

def _csv_write_rows_atomic(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=HEADER)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in HEADER})
    tmp.replace(path)

def _csv_add_record_direct(rec: dict) -> dict:
    """
    Client-side add (no MCP). Mirrors sense_add: dedupe by (term,category,locale,era),
    create file if needed, write header if new.
    Returns {"status":"added"|"exists", "file": "...", "record": rec}
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

    key = (record["term"].lower(), record["category"], record["locale"].lower(), record["era"].lower())
    if any((r.get("term","").lower(),
            r.get("category","").lower(),
            r.get("locale","").lower(),
            r.get("era","").lower()) == key for r in rows):
        return {"status": "exists", "file": str(path), "record": record}

    rows.append(record)
    _csv_write_rows_atomic(path, rows)
    return {"status": "added", "file": str(path), "record": record}


def sha256_file(p: Path) -> str:
    """
    Calculate the SHA-256 hash of a file.

    This function reads a file in chunks and calculates its SHA-256 hash.
    It is designed for efficiently handling large files by processing them
    piece by piece.

    :param p: The path to the file for which the SHA-256 hash should be
        calculated.
    :type p: Path
    :return: The hexadecimal SHA-256 hash of the file content.
    :rtype: str
    """
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

def load_state() -> Dict:
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"files": {}}

def save_state(st: Dict):
    STATE_PATH.write_text(json.dumps(st, ensure_ascii=False, indent=2), encoding="utf-8")


# ------------------ PDF text extraction ------------------
def pdf_to_texts(pdf_path: Path, ocr: bool = True, ocr_lang: str = "eng") -> List[str]:
    """
    Converts a PDF file into a list of text strings, where each string corresponds
    to the extracted text of a PDF page. Optionally supports OCR for pages with
    low text extraction reliability.

    :param pdf_path: The file path to the PDF document. This is used as an input
        to extract text from each PDF page.
    :type pdf_path: Path
    :param ocr: Boolean flag indicating whether to perform Optical Character
        Recognition (OCR) on pages where text extraction results are insufficient.
        Defaults to True.
    :type ocr: bool
    :param ocr_lang: Language setting for OCR, following the codes used by
        Tesseract OCR (e.g., "eng" for English). Defaults to "eng".
    :type ocr_lang: str
    :return: A list of strings, where each string contains the extracted or OCR-processed
        text from a corresponding page in the PDF.
    :rtype: List[str]
    """
    logger = logging.getLogger("sense-ingest")
    texts: List[str] = []
    try:
        import pdfplumber
    except Exception as e:
        logger.error("pdfplumber missing: %s", e)
        return []

    with log_timing(logger, f"open {pdf_path.name}"):
        with pdfplumber.open(str(pdf_path)) as pdf:
            n_pages = len(pdf.pages)
            logger.info("Pages: %d", n_pages)
            for i, page in enumerate(pdf.pages, 1):
                t = (page.extract_text() or "").strip()
                if t:
                    logger.debug("Page %d extracted %d chars", i, len(t))
                if ocr and len(t) < 40:
                    try:
                        from pdf2image import convert_from_path
                        import pytesseract
                        logger.debug("Page %d low text → OCR...", i)
                        img = convert_from_path(str(pdf_path), first_page=i, last_page=i)[0]
                        ot = pytesseract.image_to_string(img, lang=ocr_lang) or ""
                        t = (ot.strip() or t)
                        logger.debug("Page %d OCR result %d chars", i, len(t))
                    except Exception as e:
                        logger.warning("OCR failed on page %d: %s", i, e)
                texts.append(t)
    return texts

# ------------------ optional NER hints ------------------
def ner_hints(text: str) -> Dict[str, List[str]]:
    """
    Extracts named entities of specific types from a given text and organizes them into
    a dictionary. This function uses the spaCy library to process the text and identify
    entities categorized as "GPE", "LOC", and "DATE". If spaCy is not installed or an
    error occurs during execution, the function gracefully handles the exception and
    returns an empty dictionary for the specified keys.

    :param text: Input text to extract named entities from.
    :type text: str

    :return: A dictionary with entity types ("GPE", "LOC", "DATE") as keys and lists of
        corresponding entities as values.
    :rtype: Dict[str, List[str]]
    """
    out = {"GPE": [], "LOC": [], "DATE": []}
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text)
        for ent in doc.ents:
            if ent.label_ in out:
                out[ent.label_].append(ent.text)
    except Exception:
        pass
    return out

# ------------------ rules mapping ------------------
def load_rules(path: Optional[str]) -> Dict:
    """
    Load rules from a YAML file located at the specified path.

    This function reads YAML configuration data from the file at the given path.
    If the path is not provided or the file does not exist, an empty dictionary
    is returned. If there is an error during the file-reading or parsing process,
    a warning message is printed to standard error, and an empty dictionary is returned.

    :param path: Optional file path to the YAML configuration file.
    :type path: Optional[str]
    :return: A dictionary containing the configuration rules loaded from the file,
        or an empty dictionary if the path is not specified, the file does not exist,
        or an error occurs while reading/parsing the file.
    :rtype: Dict
    """
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        print(f"[WARN] rules file not found: {p}", file=sys.stderr)
        return {}
    try:
        return yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except Exception as e:
        print(f"[WARN] failed to read rules: {e}", file=sys.stderr)
        return {}

def apply_rules(text: str, rules: Dict) -> Dict:
    """
    Applies a set of matching rules to determine inferred categories from the given text.

    The function processes the input text and applies a series of regular expression-based
    rules provided in the `rules` dictionary. These rules are used to identify and label
    certain attributes, such as "locale," "era," "register," or "weather," in the text.
    The results are stored in a dictionary, mapping specific keys to inferred labels.

    :param text: The input text to be processed and matched against the rules.
    :type text: str
    :param rules: A dictionary containing sets of label-to-pattern mapping for various
        categories (e.g., locales, eras, registers, etc.) to guide the matching logic.
    :type rules: Dict
    :return: A dictionary containing inferred category labels and their corresponding
        matching results based on the input rules.
    :rtype: Dict
    """
    inferred = {}
    def match_block(block: Dict, target_key: str):
        if not block: return
        for label, pats in block.items():
            for pat in pats or []:
                try:
                    if re.search(pat, text, flags=re.I):
                        inferred[target_key] = label
                        return
                except re.error:
                    continue

    match_block(rules.get("locales"), "locale")
    match_block(rules.get("eras"), "era")
    match_block(rules.get("registers"), "register")
    match_block(rules.get("weather"), "weather")
    return inferred

# ------------------ OpenAI-compatible call ------------------
def chat_json(system: str, user: str, model: str = MODEL_ID, temperature: float = 0.2) -> Dict:
    """
    Creates and sends a chat completion request to the OpenAI API and processes
    the JSON response. The function utilizes the specified model, system message,
    user input, and optional temperature for generating responses.

    :param system: The system message defining the context or behavior of the
                   model.
    :type system: str
    :param user: The user input or message to which the model responds.
    :type user: str
    :param model: The identifier of the model to use. Defaults to MODEL_ID.
    :type model: str
    :param temperature: The sampling temperature. Controls the variability of
                        the model's responses. A lower value results in more
                        deterministic output. Defaults to 0.2.
    :type temperature: float
    :return: Parsed JSON object from the model's top choice containing
             the generated message.
    :rtype: Dict
    """
    logger = logging.getLogger("sense-ingest")
    url = f"{OPENAI_API_BASE.rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    payload = {
        "model": model,
        "messages": [{"role":"system","content": system},{"role":"user","content": user}],
        "response_format": {"type": "json_object"},
        "temperature": temperature,
    }
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("POST %s model=%s user_len=%d", url, model, len(user))

    t0 = time.perf_counter()
    r = requests.post(url, headers=headers, json=payload, timeout=120)
    dt = time.perf_counter() - t0
    logger.info("LLM call %.2fs", dt)
    r.raise_for_status()
    content = r.json()["choices"][0]["message"]["content"]
    m = re.search(r"\{.*}", content, flags=re.S)
    if m: content = m.group(0)
    return json.loads(content)

def extract_items(chunk: str, ctx: Dict) -> List[Dict]:
    """
    Extracts sensory-related items from a given text chunk using a provided context.

    This function processes a text chunk and extracts structured information
    about terms that fall into specific sensory categories. It utilizes a context
    for customizing the behavior of the extraction process. The extracted items
    are then filtered and cleaned for relevancy before being returned.

    :param chunk: The input text content to process.
    :param ctx: A dictionary containing contextual parameters relevant for
        extraction.
    :return: A list of dictionaries, where each dictionary represents a sensory
        term, its category, and associated notes.
    """
    logger = logging.getLogger("sense-ingest")
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("Extract with ctx: %s", {k: ctx.get(k) for k in ("locale","era","register","weather")})
    if logging.getLogger("sense-ingest").isEnabledFor(logging.DEBUG):
        logger.debug("Chunk preview: %s", (chunk[:160] + "…") if len(chunk) > 160 else chunk)
    obj = chat_json(SYSTEM, USER_TMPL.format(
        locale=ctx.get("locale",""), era=ctx.get("era",""),
        register=ctx.get("register",""), weather=ctx.get("weather",""),
        passage=chunk
    ))
    items = obj.get("items", []) if isinstance(obj, dict) else []
    cleaned = []
    for it in items:
        term = (it.get("term","") or "").strip()
        cat  = (it.get("category","") or "").strip().lower()
        notes= (it.get("notes","") or "").strip()
        if term and cat in {"smell","sound","taste","touch","sight"}:
            cleaned.append({"term": term, "category": cat, "notes": notes})
    logger.info("Extracted %d items", len(cleaned))
    return cleaned


# ------------------ extraction LLM prompt ------------------
SYSTEM = """You extract *explicit* sensory details from historical/literary passages.

Return strict JSON: {"items":[
  {"term": "...", "category": "smell|sound|taste|touch|sight", "notes": "..."}
]}

Rules:
- Only include cues stated or strongly implied; do not invent.
- "term" short + reusable; "notes" <= 12 words.
- Use exactly one of: smell, sound, taste, touch, sight.
- Avoid anachronisms; defer to provided CONTEXT if consistent with the text.
- 0..12 items per chunk is fine.
"""

USER_TMPL = """Extract sensory items from the passage. Output JSON only.

CONTEXT (may be incomplete):
- Locale: {locale}
- Era: {era}
- Register: {register}
- Weather/Time: {weather}

PASSAGE:
{passage}
"""

def _ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def _safe_rel(parent: Path, child: Path) -> str:
    try:
        return str(child.relative_to(parent))
    except Exception:
        return str(child)

def _summarize_cats(d: Dict[str, int]) -> str:
    """
    Summarizes the given categories in a predefined order and includes any additional
    categories not defined in the order. The resulting summary is a comma-separated
    string with each category and its associated value.

    :param d: Dictionary of categories (str) with their associated integer values.
    :type d: Dict[str, int]
    :return: A string summarizing the categories and their values.
    :rtype: str
    """
    if not d: return "-"
    order = ["smell","sound","taste","touch","sight"]
    parts = [f"{k}:{d.get(k,0)}" for k in order if k in d or d.get(k,0)]
    # add any other stray categories
    for k,v in d.items():
        if k not in order:
            parts.append(f"{k}:{v}")
    return ", ".join(parts) if parts else "-"

def _render_report_md(stats: Dict, moved_to: list[str], final_dir: Optional[Path]) -> str:
    """
    Generates a Markdown string representing a detailed report of the ingest statistics.

    The function processes given statistics, including file details, processing statuses,
    defaults, and options. It also includes a summary of categorized data, page-specific
    information, and directories processed during the operation.

    :param stats: A dictionary containing ingest statistics. The expected keys include:
                  - 'file': The name of the file being processed.
                  - 'sha256': Optional checksum of the file.
                  - 'total': Total number of records processed.
                  - 'added': Number of records added.
                  - 'exists': Number of records that already exist.
                  - 'defaults': A dictionary of default metadata like locale, era, register,
                    and weather.
                  - 'per_category': A dictionary summarizing totals by category.
                  - 'options': A dictionary of process options such as OCR, dry run status,
                    and named entity recognition.
                  - 'sidecar': A boolean indicating if a sidecar file is present.
                  - 'rules': An optional string indicating a rules file applied.
                  - 'ok': A boolean indicating if the operation succeeded.
                  - 'pages': A list of page-specific details, such as:
                    - 'pageno': Page number.
                    - 'locale', 'era', 'register', 'weather': Metadata for each page.
                    - 'items': Total items found on the page.
                    - 'examples': List of examples found on the page (up to 5 shown).
                    - 'by_category': Summary totals by category per page.
    :param moved_to: A list of directories or locations where files were moved.
    :param final_dir: An optional Path object representing the final directory's location.

    :return: A string in markdown format representing the ingest report.
    """
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    per_cat = _summarize_cats(stats.get("per_category", {}))
    defaults = stats.get("defaults", {})
    options  = stats.get("options", {})
    sidecar  = stats.get("sidecar", False)
    rules    = stats.get("rules")
    ok       = stats.get("ok", True)
    status   = "ok" if ok and not options.get("dry_run") else ("dry-run" if options.get("dry_run") else "error")
    lines = []
    lines.append(f"# Ingest Report — {stats['file']}")
    lines.append("")
    lines.append(f"- **Generated:** {stamp}")
    lines.append(f"- **SHA256:** `{stats.get('sha256','')}`")
    lines.append(f"- **Status:** **{status}**")
    lines.append(f"- **Totals:** total={stats.get('total',0)}, added={stats.get('added',0)}, exists={stats.get('exists',0)}")
    lines.append(f"- **By category:** {per_cat}")
    lines.append(f"- **Defaults:** locale={defaults.get('locale','')}, era={defaults.get('era','')}, register={defaults.get('register','')}, weather={defaults.get('weather','')}")
    lines.append(f"- **Sidecar present:** {bool(sidecar)}")
    lines.append(f"- **Rules file:** {rules or '-'}")
    lines.append(f"- **Options:** ocr={options.get('ocr')}, ocr_lang={options.get('ocr_lang')}, ner={options.get('ner', None)}, dry_run={options.get('dry_run')}")
    if moved_to:
        lines.append(f"- **Moved to:**")
        for m in moved_to:
            lines.append(f"  - `{m}`")
    if final_dir:
        lines.append(f"- **Final directory:** `{final_dir}`")
    lines.append("")
    # Pages table
    pages = stats.get("pages", [])
    if pages:
        lines.append("## Pages")
        lines.append("| page | locale | era | register | weather | items | examples | by_category |")
        lines.append("|---:|---|---|---|---|---:|---|---|")
        for pg in pages:
            ex = ", ".join(pg.get("examples", [])[:5])
            bc = _summarize_cats(pg.get("by_category", {}))
            lines.append(f"| {pg.get('pageno','')} | {pg.get('locale','')} | {pg.get('era','')} | {pg.get('register','')} | {pg.get('weather','')} | {pg.get('items',0)} | {ex} | {bc} |")
        lines.append("")
    return "\n".join(lines)

def _write_report_files(report_dir: Path, stats: Dict, moved_to: list[str], final_dir: Optional[Path], fmt: str):
    """
    Writes report files in specified formats (Markdown, JSON, or both) to a specified subdirectory.
    Each generated report file is tagged with a timestamp. Optionally, pointers to the most recently
    generated reports are written for quick access.

    The function creates the necessary subdirectories before writing the files to ensure
    proper file organization.

    :param report_dir: The base directory to store the generated report files.
    :type report_dir: Path
    :param stats: The statistics or data to include in the reports.
    :type stats: Dict
    :param moved_to: List of filenames or paths that the process moved during its execution.
    :type moved_to: list[str]
    :param final_dir: The final directory where the process output resides. Can be None.
    :type final_dir: Optional[Path]
    :param fmt: The format(s) of the report to generate. Accepts 'md', 'json', or 'both'.
    :type fmt: str
    :return: A tuple containing the subdirectory where reports are stored and a list of generated file paths.
    :rtype: Tuple[Path, List[Path]]
    """
    _ensure_dir(report_dir)
    stem = Path(stats["file"]).stem
    subdir = report_dir / stem
    _ensure_dir(subdir)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    written = []

    if fmt in ("md","both"):
        md = _render_report_md(stats, moved_to, final_dir)
        md_path = subdir / f"{ts}.md"
        md_path.write_text(md, encoding="utf-8")
        # keep a 'latest.md' pointer
        (subdir / "latest.md").write_text(md, encoding="utf-8")
        written.append(md_path)

    if fmt in ("json","both"):
        js = dict(stats)
        js["moved_to"] = moved_to
        js["final_dir"] = str(final_dir) if final_dir else None
        js["generated_at"] = datetime.now().isoformat()
        js_path = subdir / f"{ts}.json"
        js_path.write_text(json.dumps(js, ensure_ascii=False, indent=2), encoding="utf-8")
        (subdir / "latest.json").write_text(json.dumps(js, ensure_ascii=False, indent=2), encoding="utf-8")
        written.append(js_path)

    return subdir, written

def _update_index(report_dir: Path, stats: Dict, subdir: Path, written: list[Path]):
    """
    Updates the index file in the given report directory with statistics about
    the current processing and optionally a link to the latest report in the subdirectory.

    The method ensures that the directory exists, writes a new table header if the
    index file does not already exist, and appends a formatted line of statistics
    with links to relevant reports. This functionality is essential for maintaining an
    overview of processing results.

    :param report_dir: The directory where the index file should be updated.
    :type report_dir: Path
    :param stats: A dictionary containing statistics about the operation
                  (e.g., file name, total, added, exists, processing status).
    :type stats: Dict
    :param subdir: The subdirectory containing processed files and reports.
    :type subdir: Path
    :param written: A list of paths of written report files during the current operation.
    :type written: list[Path]
    :return: None
    """
    idx = report_dir / "index.md"
    _ensure_dir(report_dir)
    if not idx.exists():
        idx.write_text("# Ingest Index\n\n| date | file | status | total | added | exists | by_category | report |\n|---|---|---|---:|---:|---:|---|---|\n", encoding="utf-8")

    date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    status = "ok" if stats.get("ok", True) and not stats.get("options",{}).get("dry_run") else ("dry-run" if stats.get("options",{}).get("dry_run") else "error")
    cats = _summarize_cats(stats.get("per_category", {}))
    # link to latest.md (exists for md/both), else first written file
    link_target = subdir / ("latest.md" if (subdir / "latest.md").exists() else (written[0].name if written else ""))
    link = _safe_rel(report_dir, link_target)
    line = f"| {date} | {stats['file']} | {status} | {stats.get('total',0)} | {stats.get('added',0)} | {stats.get('exists',0)} | {cats} | [{Path(link).name}]({link}) |\n"
    with idx.open("a", encoding="utf-8") as f:
        f.write(line)

def _dest_dir(base: Path, group_by_date: bool) -> Path:
    if group_by_date:
        return base / datetime.now().strftime("%Y%m%d")
    return base

def _safe_move(src: Path, dst_dir: Path) -> Path:
    dst_dir.mkdir(parents=True, exist_ok=True)
    target = dst_dir / src.name
    if target.exists():
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        target = dst_dir / f"{src.stem}_{stamp}{src.suffix}"
    return Path(shutil.move(str(src), str(target)))

def _move_file_group(pdf: Path, dst_dir: Path) -> list[str]:
    """
    Moves a group of files related to a specified PDF to a destination directory. This function moves
    the PDF file itself as well as any associated YAML or YML files found with the same base name.

    :param pdf: The path to the PDF file that needs to be moved. This may also serve as the base
                name for associated YAML or YML files.
    :type pdf: Path
    :param dst_dir: The destination directory where the files will be moved. Must be a valid
                    directory path.
    :type dst_dir: Path
    :return: A list of strings representing the paths of all files successfully moved to the
             destination directory.
    :rtype: list[str]
    """
    moved = []
    if pdf.exists():
        moved.append(str(_safe_move(pdf, dst_dir)))
    for ext in (".yml", ".yaml"):
        side = pdf.with_suffix(ext)
        if side.exists():
            moved.append(str(_safe_move(side, dst_dir)))
    return moved


# Update your sense_add function with better timeout handling:

def sense_add(record: Dict) -> Dict:
    """
    Wrapper around mcp_call('sense_add', ...) with better diagnostics and timeout.
    """
    import time, json, logging

    log = logging.getLogger("sense-ingest")

    def _short(d: Dict) -> str:
        # compact preview for logs
        view = {
            "term": d.get("term"),
            "category": d.get("category"),
            "locale": d.get("locale"),
            "era": d.get("era"),
            "weather": d.get("weather"),
            "register": d.get("register"),
            "notes": (d.get("notes") or "")
        }
        if len(view["notes"]) > 80:
            view["notes"] = view["notes"][:77] + "…"
        return json.dumps(view, ensure_ascii=False)

    # --- 1) Validate minimal shape
    required = ("term", "category", "locale", "era")
    missing = [k for k in required if not str(record.get(k, "")).strip()]
    if missing:
        msg = f"missing required: {', '.join(missing)}"
        log.error("sense_add invalid record: %s | %s", msg, _short(record))
        return {"status": "error", "error": msg, "record": record}

    # --- 2) Normalize values
    rec = dict(record)
    rec["term"] = str(rec["term"]).strip()
    rec["category"] = str(rec.get("category", "")).strip().lower()
    rec["locale"] = str(rec.get("locale", "")).strip()
    rec["era"] = str(rec.get("era", "")).strip()
    rec["weather"] = (str(rec.get("weather", "")).strip().lower() or "any")
    rec["register"] = (str(rec.get("register", "")).strip().lower() or "common")

    # --- 3) Retry loop with longer timeout
    retries, delay = 2, 0.25
    last: Dict[str, Any] = {"status": "error", "error": "no attempt"}

    for attempt in range(1, retries + 2):
        # Use longer timeout for each attempt, escalating
        timeout = 15.0 + (attempt * 10.0)  # 15s, 25s, 35s

        log.debug(f"sense_add attempt {attempt}/{retries + 1} with {timeout}s timeout: {rec.get('term')}")

        last = mcp_call("sense_add", rec, timeout=timeout)
        status = last.get("status")

        if status in ("added", "exists"):
            if attempt > 1:
                log.info("sense_add OK after retry x%d: %s", attempt - 1, record.get("term"))
            return last

        # Log the issue but continue retrying
        if attempt == 1:
            log.warning("MCP write issue (attempt %d/%d) term=%r → %s",
                        attempt, retries + 1, record.get("term"),
                        json.dumps(last, ensure_ascii=False)[:200])

        time.sleep(delay)
        delay *= 1.5  # Exponential backoff

    log.error("sense_add failed after %d attempts: term=%r → %s",
              retries + 1, record.get("term"), json.dumps(last, ensure_ascii=False)[:200])
    return last

# ------------------ sidecar (.yml) ------------------
def sidecar_for(pdf: Path) -> Optional[Path]:
    for ext in (".yml",".yaml"):
        p = pdf.with_suffix(ext)
        if p.exists(): return p
    return None

def load_sidecar(pdf: Path) -> Dict:
    p = sidecar_for(pdf)
    if not p: return {}
    try:
        return yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}

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

@contextmanager
def log_timing(logger: logging.Logger, label: str, level=logging.DEBUG):
    t0 = time.perf_counter()
    logger.log(level, f"▶ {label}")
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        logger.log(level, f"✓ {label} (%.2fs)", dt)


# ------------------ main pipeline ------------------
def infer_context(page_text: str, rules: dict | None, defaults: dict, use_ner: bool = True) -> dict:
    """
    Infer {locale, era, register, weather} from a page of text.
    - Start with defaults
    - Apply YAML rules if available (safe fallback)
    - If locale/era still missing and use_ner=True, try NER hints
    - Fill remaining gaps with simple heuristics
    """
    import re, logging
    logger = logging.getLogger("sense-ingest")

    t = page_text or ""
    ctx = {
        "locale":  (defaults.get("locale") or "").strip(),
        "era":     (defaults.get("era") or "").strip(),
        "register":(defaults.get("register") or "common").strip(),
        "weather": (defaults.get("weather") or "any").strip(),
    }
    sources: list[str] = ["defaults"]

    # ---------- 1) Rules (with safe fallback) ----------
    def _apply_rules_fallback(text: str, rulebook: dict | None) -> dict:
        out: dict = {}
        if not rulebook:
            return out

        def _pick(section: str, key: str):
            section_map = rulebook.get(section, {}) or {}
            for label, patterns in section_map.items():
                for pat in (patterns or []):
                    try:
                        if re.search(pat, text, re.I):
                            out[key] = label
                            return
                    except re.error:
                        # Ignore malformed regex in YAML
                        continue

        _pick("locales",   "locale")
        _pick("eras",      "era")
        _pick("registers", "register")
        _pick("weather",   "weather")
        return out

    inferred = {}
    try:
        if rules:
            # If your project provides apply_rules, use it:
            inferred = apply_rules(t, rules)  # type: ignore[name-defined]
    except Exception:
        inferred = _apply_rules_fallback(t, rules)

    if inferred:
        for k, v in inferred.items():
            if v:
                ctx[k] = str(v).strip()
        sources.append("rules")

    # ---------- 2) NER hints (only if locale/era still missing) ----------
    if use_ner and (not ctx["locale"] or not ctx["era"]):
        try:
            hints = ner_hints(t[:2000])  # type: ignore[name-defined]
        except Exception:
            hints = {}

        gpes  = " ".join(hints.get("GPE",  []))
        dates = " ".join(hints.get("DATE", []))
        hay   = f"{t} {gpes} {dates}"

        before_loc, before_era = ctx["locale"], ctx["era"]

        if not ctx["locale"]:
            if re.search(r"\b(Kyoto|Kyōto|Nara|Heian|Yamato|Japan)\b", hay, re.I):
                ctx["locale"] = "Japan"
            elif re.search(r"\b(Baghdad|Tigris|Samarra)\b", hay, re.I):
                ctx["locale"] = "Abbasid Baghdad"
            elif re.search(r"\b(Venice|Venezia|Rialto|Lagoon)\b", hay, re.I):
                ctx["locale"] = "Venice"
            elif re.search(r"\b(Cusco|Cuzco|Andes|Quito|Titicaca|Chim[úu]|Moche)\b", hay, re.I):
                ctx["locale"] = "Andes"

        if not ctx["era"]:
            if re.search(r"\bHeian\b|\b(79[4-9]|8\d\d|11[0-7]\d)\b", hay, re.I):
                ctx["era"] = "Heian"
            elif re.search(r"\bSong\b", hay, re.I):
                ctx["era"] = "Song"
            elif re.search(r"\bMughal\b", hay, re.I):
                ctx["era"] = "Mughal"
            elif re.search(r"\bChim[úu]\b|\bChim[úu]\s*Empire\b", hay, re.I):
                ctx["era"] = "Chimú"

        if ctx["locale"] != before_loc or ctx["era"] != before_era:
            sources.append("ner")

    # ---------- 3) Heuristics for register/weather ----------
    tl = t.lower()
    before_reg, before_wx = ctx["register"], ctx["weather"]

    if not ctx["register"] or ctx["register"].lower() == "common":
        if re.search(r"\b(palace|chamberlain|courtier|imperial|empress|retainers?)\b", tl):
            ctx["register"] = "court"
        elif re.search(r"\b(quay|rigging|tar|harbour|harbor|deck|keel|mast|bilge|sail)\b", tl):
            ctx["register"] = "maritime"
        elif re.search(r"\b(monastery|temple|abbot|sutra|monk|bell|cloister)\b", tl):
            ctx["register"] = "monastic"
        elif re.search(r"\b(festival|procession|drum|lantern|bonfire|parade)\b", tl):
            ctx["register"] = "festival"
        elif re.search(r"\b(market|bazaar|vendor|stall|hawker|merchant)\b", tl):
            ctx["register"] = "market"
        elif re.search(r"\b(garrison|encampment|spear|banner|shield|lance|archer)\b", tl):
            ctx["register"] = "military"
        elif re.search(r"\b(scroll|inkstone|calligraphy|study|classic|scholar)\b", tl):
            ctx["register"] = "scholar"
        elif re.search(r"\b(garden|moss|gravel|raked|lantern|pond|koi)\b", tl):
            ctx["register"] = "garden"

    if not ctx["weather"] or ctx["weather"].lower() == "any":
        if re.search(r"\bmonsoon\b", tl):
            ctx["weather"] = "monsoon"
        elif re.search(r"\bdry\s+season\b", tl):
            ctx["weather"] = "dry season"
        elif re.search(r"\bwinter\s+snow\b|\bsnowfall\b|\bsnow\b", tl):
            ctx["weather"] = "winter snow"
        elif re.search(r"\brain(fall)?|downpour|storm|shower|drizzle\b", tl):
            ctx["weather"] = "rain"
        elif re.search(r"\bdawn|daybreak|sunrise\b", tl):
            ctx["weather"] = "dawn"
        elif re.search(r"\bnight|moonlit|midnight|nightfall\b", tl):
            ctx["weather"] = "night"
        elif re.search(r"\bsummer\b", tl):
            ctx["weather"] = "summer"
        elif re.search(r"\bwinter\b", tl):
            ctx["weather"] = "winter"

    if ctx["register"] != before_reg or ctx["weather"] != before_wx:
        sources.append("heuristics")

    # ---------- 4) Normalize enums ----------
    def norm_space(s: str) -> str:
        return " ".join((s or "").split())

    ctx["locale"]   = norm_space(ctx["locale"])
    ctx["era"]      = norm_space(ctx["era"])
    ctx["register"] = (norm_space(ctx["register"]) or "common").lower()
    ctx["weather"]  = (norm_space(ctx["weather"])  or "any").lower()

    logger.debug("Context %s via %s", ctx, ", ".join(sources))
    return ctx




def ingest_pdf(pdf_path: Path, args) -> Dict:
    """
    Processes a given PDF file to extract structured information and metadata.

    The function ingests a PDF file, processes its content using OCR (if enabled),
    rules, and other contextual settings. It deduplicates extracted items and
    categorizes them before returning a comprehensive summary, including statistics.

    :param pdf_path: Path to the PDF file to process.
    :type pdf_path: Path
    :param args: Arguments containing configuration options, such as rules,
        language settings, and processing flags.
    :type args: Any
    :return: A dictionary containing the summary of processed items and metadata,
        including the deduplicated items, statistics on their categories, page-level
        summaries, processing options, and ingestion status.
    :rtype: Dict
    """
    t0 = time.perf_counter()
    logger = logging.getLogger("sense-ingest")
    logger.info("▶ Start ingest: %s", pdf_path.name)

    side = load_sidecar(pdf_path)
    rules = load_rules(getattr(args, "rules", None))
    defaults = {
        "locale": side.get("locale") or args.default_locale or "",
        "era": side.get("era") or args.default_era or "",
        "register": side.get("register") or args.default_register or DEFAULT_REGISTER,
        "weather": side.get("weather") or args.default_weather or DEFAULT_WEATHER,
    }
    use_ner = not getattr(args, "no_ner", False)

    t0 = time.perf_counter()
    page_texts = pdf_to_texts(pdf_path, ocr=not args.no_ocr, ocr_lang=args.ocr_lang)
    n_pages = len(page_texts)
    logger.info("Pages: %d (ocr=%s, lang=%s)", n_pages, not args.no_ocr, args.ocr_lang)

    pages_summary: List[Dict] = []
    all_items: List[Dict] = []

    # heartbeat config
    hb = max(getattr(args, "heartbeat", 15), 1)
    last_hb = time.perf_counter()

    for pageno, txt in enumerate(page_texts, 1):
        if not txt.strip():
            logger.debug("Page %d empty, skipping", pageno)
            continue

        # heartbeat
        now = time.perf_counter()
        if now - last_hb >= hb:
            logger.info("… working (%d/%d)", pageno, n_pages)
            last_hb = now

        # infer context
        if "infer_context" in globals():
            ctx = infer_context(txt, rules, defaults, use_ner)
        else:
            ctx = infer_context(txt, rules, defaults)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Page %d ctx: %s", pageno, {k: ctx.get(k) for k in ("locale","era","register","weather")})

        # extract items with timing
        t_llm = time.perf_counter()
        items = extract_items(txt, ctx)
        dt_llm = time.perf_counter() - t_llm
        logger.info("Page %d/%d: %d items (LLM %.2fs)", pageno, n_pages, len(items), dt_llm)
        if logger.isEnabledFor(logging.DEBUG) and items:
            logger.debug("Examples: %s", ", ".join(it["term"] for it in items[:5]))

        # collect rows
        all_items.extend([
            {
                "term": it["term"], "category": it["category"],
                "notes": (it.get("notes","") + f" (src: {pdf_path.name} p.{pageno})").strip(),
                "locale": ctx.get("locale") or defaults["locale"],
                "era": ctx.get("era") or defaults["era"],
                "register": ctx.get("register") or defaults["register"],
                "weather": ctx.get("weather") or defaults["weather"],
            }
            for it in items
        ])

        # page summary
        by_cat = Counter([it["category"] for it in items])
        pages_summary.append({
            "pageno": pageno,
            "locale": ctx.get("locale") or "",
            "era": ctx.get("era") or "",
            "register": ctx.get("register") or "",
            "weather": ctx.get("weather") or "",
            "items": len(items),
            "examples": [it["term"] for it in items[:5]],
            "by_category": dict(by_cat),
        })

    # dedupe by (term, category, locale, era)
    seen = set()
    uniq: List[Dict] = []
    for rec in all_items:
        key = (rec["term"].lower(), rec["category"], rec["locale"].lower(), rec["era"].lower())
        if key not in seen:
            seen.add(key)
            uniq.append(rec)

    per_cat = Counter([r["category"] for r in uniq])
    logger.info("Deduped to %d unique items (%s)", len(uniq), _summarize_cats(dict(per_cat)))

    stats = {
        "file": pdf_path.name,
        "sha256": sha256_file(pdf_path),
        "total": len(uniq),
        "added": 0, "exists": 0,
        "per_category": dict(per_cat),
        "pages": pages_summary,
        "defaults": defaults,
        "sidecar": bool(side),
        "rules": getattr(args, "rules", None),
        "options": {
            "ocr": not args.no_ocr,
            "ocr_lang": args.ocr_lang,
            "ner": use_ner,
            "dry_run": args.dry_run
        },
        "ok": True,
    }

    if args.dry_run:
        logger.info("[dry-run] %s: total=%d (%s)", pdf_path.name, stats["total"], _summarize_cats(stats["per_category"]))
        logger.info("✓ Ingest finished in %.2fs (dry-run)", time.perf_counter() - t0)
        return stats

    # write via MCP with progress logs
    logger.info("Writing %d items via MCP…", len(uniq))
    write_stats = {"added": 0, "exists": 0, "error": 0}
    if args.bypass_mcp:
        for rec in uniq:
            res = _csv_add_record_direct(rec)
            st = res.get("status")
            if st == "added":
                write_stats["added"] += 1
            elif st == "exists":
                write_stats["exists"] += 1
            else:
                write_stats["error"] += 1
    else:
        for rec in uniq:
            res = sense_add(rec)  # MCP path
            st = res.get("status")
            if st == "added":
                write_stats["added"] += 1
            elif st == "exists":
                write_stats["exists"] += 1
            else:
                write_stats["error"] += 1
                # optional: log the anomaly
                logging.getLogger("sense-ingest").warning("Write anomaly on %r: %s", rec.get("term"), res)

    stats["added"] = write_stats["added"]
    stats["exists"] = write_stats["exists"]
    stats["write_stats"] = write_stats

    logger.info(
        "✓ Ingest finished in %.2fs (added=%d, exists=%d, errors=%d, total=%d)",
        time.perf_counter() - t0,
        write_stats["added"], write_stats["exists"], write_stats["error"],
        len(uniq),
    )
    return stats



def _is_file_stable(p: Path, wait: float = 0.5, checks: int = 4) -> bool:
    """
    Treat a file as 'ready' only if its size+mtime stop changing.
    This avoids ingesting while a copy is still in progress.
    """
    try:
        last = None
        for _ in range(checks):
            st = p.stat()
            cur = (st.st_size, st.st_mtime_ns)
            if cur == last:
                return True
            last = cur
            time.sleep(wait)
    except FileNotFoundError:
        return False
    return False


def _list_pdfs(d: Path) -> list[Path]:
    """
    Return non-hidden PDFs in d (case-insensitive on extension), skipping temporary/marker files.
    """
    # Use a case-insensitive glob
    pdfs = set()
    for pat in ("*.pdf", "*.PDF", "*.[Pp][Dd][Ff]"):
        pdfs.update(d.glob(pat))

    out = []
    for p in pdfs:
        name = p.name
        if name.startswith("."):              # hidden
            continue
        if name.endswith(".processing"):      # our in-progress marker (if you add it)
            continue
        # sometimes weird extensions like '.pdf ' exist; normalize by rstrip
        if not name.rstrip().lower().endswith(".pdf"):
            continue
        out.append(p)

    # deterministic order
    return sorted(out, key=lambda x: x.name.lower())


def watch_dir(docs_dir: Path, args):
    logger = logging.getLogger("sense-ingest")
    logger.info("[watch] %s (Ctrl+C to stop)", docs_dir.resolve())

    state = load_state()

    hb = max(getattr(args, "heartbeat", 15), 1)
    # Make the first heartbeat fire immediately
    last_beat = time.time() - hb

    try:
        while True:
            pdfs = _list_pdfs(docs_dir)

            # Debug: show what we see every loop
            logger.debug("Scan %s → %d candidates", docs_dir.resolve(), len(pdfs))
            if pdfs:
                logger.debug("Candidates: %s", ", ".join(p.name for p in pdfs))

            # Idle heartbeat
            now = time.time()
            if not pdfs and now - last_beat >= hb:
                logger.info("… watching (no PDFs). Poll again in %ds", args.poll)
                last_beat = now

            for p in pdfs:
                # Quick unchanged check via size/mtime (fast)
                rec = state["files"].get(str(p))
                try:
                    st = p.stat()
                except FileNotFoundError:
                    continue  # disappeared
                if rec and args.only_new and rec.get("size") == st.st_size and rec.get("mtime_ns") == st.st_mtime_ns:
                    logger.debug("Skip unchanged (size/mtime): %s", p.name)
                    continue

                # Stable file check so we don't ingest while it's copying
                if not _is_file_stable(p, wait=0.5, checks=4):
                    logger.info("Waiting for file to settle: %s", p.name)
                    continue

                # Real hash for only-new
                try:
                    h = sha256_file(p)
                except Exception as e:
                    logger.warning("Hash failed for %s: %s", p.name, e)
                    continue
                if rec and args.only_new and rec.get("sha256") == h:
                    logger.debug("Skip unchanged (sha256): %s", p.name)
                    # still refresh size/mtime in state
                    state["files"][str(p)]["size"] = st.st_size
                    state["files"][str(p)]["mtime_ns"] = st.st_mtime_ns
                    save_state(state)
                    continue

                logger.info("[ingest] %s", p.name)
                ok = True
                moved_to: list[str] = []
                final_dir: Optional[Path] = None

                try:
                    stats = ingest_pdf(p, args)
                except Exception as e:
                    ok = False
                    logger.exception("Ingest error on %s", p.name)
                    stats = {
                        "file": p.name, "sha256": h, "total": 0, "added": 0, "exists": 0,
                        "per_category": {}, "pages": [], "defaults": {}, "sidecar": False,
                        "rules": getattr(args, "rules", None),
                        "options": {"dry_run": getattr(args, "dry_run", False)},
                        "ok": False, "error": str(e),
                    }

                # Move policy
                if not getattr(args, "dry_run", False) and getattr(args, "move", "success") != "never":
                    try:
                        to_done = ok or (args.move == "always")
                        base_dir = Path(args.done_dir if to_done else args.fail_dir)
                        dst_dir = _dest_dir(base_dir, getattr(args, "group_by_date", False))
                        moved_to = _move_file_group(p, dst_dir)
                        final_dir = dst_dir
                        if moved_to:
                            logger.info("Moved %s → %s", p.name, dst_dir)
                    except Exception as e:
                        logger.warning("Move failed for %s: %s", p.name, e)

                # Reports
                try:
                    rep_dir = Path(getattr(args, "report_dir", "exports/ingest_reports"))
                    subdir, written = _write_report_files(rep_dir, stats, moved_to, final_dir, getattr(args, "report_format", "md"))
                    if getattr(args, "report_index", False):
                        _update_index(rep_dir, stats, subdir, written)
                    if written:
                        logger.info("Report: %s", written[-1])
                except Exception as e:
                    logger.warning("Report failed for %s: %s", p.name, e)

                # Persist state (store size/mtime for fast-only-new)
                state["files"][str(p)] = {
                    "sha256": h, "size": st.st_size, "mtime_ns": st.st_mtime_ns,
                    "stats": stats, "ok": ok,
                    "moved_to": moved_to, "final_dir": str(final_dir) if final_dir else None,
                    "ts": int(time.time()),
                }
                save_state(state)

            time.sleep(getattr(args, "poll", 5))
    except KeyboardInterrupt:
        logger.info("[watch] stopped")

def main():
    """CLI entrypoint for ingesting PDFs and writing Sense-Bank rows + reports."""
    import argparse, time
    from pathlib import Path

    ap = argparse.ArgumentParser(
        description="Ingest PDFs from docs/ and add sensory cues to Sense-Bank via MCP."
    )
    ap.add_argument("--docs-dir", default="docs")
    ap.add_argument("--rules")
    ap.add_argument("--default-locale")
    ap.add_argument("--default-era")
    ap.add_argument("--default-register", default=DEFAULT_REGISTER)
    ap.add_argument("--default-weather",  default=DEFAULT_WEATHER)
    ap.add_argument("--no-ocr", action="store_true")
    ap.add_argument("--ocr-lang", default="eng")
    ap.add_argument("--data-dir", default="data", help="Directory for Sense-Bank CSVs")
    ap.add_argument("--state-file", default="exports/ingest_reports/state.json")
    # use args.state_file in load_state()/save_state()
    ap.add_argument("--log-level", choices=["DEBUG","INFO","WARNING","ERROR"], default="INFO")
    ap.add_argument("--log-file", default=None)
    ap.add_argument("--heartbeat", type=int, default=15)
    ap.add_argument("--trace-llm", action="store_true")
    ap.add_argument("--done-dir", default="docs_done")
    ap.add_argument("--fail-dir", default="docs_fail")
    ap.add_argument("--move", choices=["success","always","never"], default="success")
    ap.add_argument("--bypass-mcp", action="store_true",
                    help="Write directly to CSVs instead of calling MCP")
    ap.add_argument("--report-dir", default="exports/ingest_reports")
    ap.add_argument("--report-format", choices=["md","json","both"], default="md")
    ap.add_argument("--report-index", action="store_true")
    ap.add_argument("--group-by-date", action="store_true")
    ap.add_argument("--print-json", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--only-new", action="store_true")
    ap.add_argument("--poll", type=int, default=5)
    args = ap.parse_args()

    global DATA_DIR
    DATA_DIR = Path(args.data_dir)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

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

    if args.once:
        state = load_state()
        for p in sorted(docs_dir.glob("*.pdf")):
            logger.info("[ingest] %s", p.name)
            h = sha256_file(p)
            ok = True
            moved_to: list[str] = []
            final_dir: Optional[Path] = None
            try:
                stats = ingest_pdf(p, args)
            except Exception as e:
                ok = False
                logger.exception("Ingest error on %s", p.name)
                stats = {"file": p.name, "sha256": h, "total": 0, "added": 0, "exists": 0,
                         "per_category": {}, "pages": [], "defaults": {}, "sidecar": False,
                         "rules": args.rules, "options": {"dry_run": args.dry_run}, "ok": False,
                         "error": str(e)}

            if args.print_json:
                import json; print(json.dumps(stats, ensure_ascii=False, indent=2))

            if not args.dry_run and args.move != "never":
                to_done = ok or (args.move == "always")
                base_dir = Path(args.done_dir if to_done else args.fail_dir)
                dst_dir  = _dest_dir(base_dir, args.group_by_date)
                moved_to = _move_file_group(p, dst_dir); final_dir = dst_dir
                if moved_to: logger.info("Moved %s → %s", p.name, dst_dir)

            rep_dir = Path(args.report_dir)
            subdir, written = _write_report_files(rep_dir, stats, moved_to, final_dir, args.report_format)
            if args.report_index: _update_index(rep_dir, stats, subdir, written)
            if written: logger.info("Report: %s", written[-1])

            state["files"][str(p)] = {
                "sha256": h, "stats": stats, "ok": ok,
                "moved_to": moved_to, "final_dir": str(final_dir) if final_dir else None,
                "ts": int(time.time())
            }
            save_state(state)
        return

    # watch mode
    watch_dir(docs_dir, args)

if __name__ == "__main__":
    main()
