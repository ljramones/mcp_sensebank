#!/usr/bin/env python3
"""
MCP (Model Context Protocol) client functionality.
Handles all MCP communication, session management, and tool calls.
"""

import os
import json
import time
import uuid
import logging
import atexit
import requests
import threading
from typing import Any, Dict, List, Optional
from contextlib import ExitStack


# ------------------ MCP Configuration ------------------
MCP_URL = os.getenv("MCP_URL", "http://127.0.0.1:8000/mcp").rstrip("/")
MCP_SESSION_HEADER = os.getenv("MCP_SESSION_HEADER")
_MCP_TRANSPORT_HEADER = os.getenv("MCP_TRANSPORT_HEADER", "OpenAI-Transport-Id")


# ------------------ MCP Global State ------------------
_MCP_CLIENT = None
_MCP_LOCK = threading.Lock()
_MCP_STACK = ExitStack()
_MCP_DEBUG_ONCE = True
_MCP_SESSION: Optional[requests.Session] = None
_MCP_SESSION_ID: Optional[str] = None
_MCP_INITIALIZED: bool = False

# Session/Transport header names to try
_MCP_SESSION_HEADER_NAMES = tuple(x for x in (
    os.getenv("MCP_SESSION_HEADER"),
    "OpenAI-Transport-Id", "X-OpenAI-Transport-Id",
    "OpenAI-Session-Id", "X-OpenAI-Session-Id",
    "X-MCP-Session", "X-Session",
) if x)

# Cookie keys to check for session IDs
_MCP_COOKIE_KEYS = (
    "mcp-session", "mcp_session", "sessionid", "session",
    "transport", "transport_id", "openai_transport_id"
)

# SSE endpoints to try
_MCP_SSE_PATHS = tuple(
    p for p in (
        os.getenv("MCP_SSE_PATH"),
        "", "/", "/events", "/stream", "/sse"
    ) if p is not None
)

_MCP_SESSION_PARAM_NAMES = ("sessionId", "transportId")


# ------------------ Session Management ------------------
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


def _id_from_any_response(resp: requests.Response) -> Optional[str]:
    """Extract session/transport ID from any response."""
    # 1) headers: accept any header whose name mentions session/transport/mcp
    for k, v in resp.headers.items():
        kl = k.lower()
        if ("session" in kl) or ("transport" in kl) or ("mcp" in kl):
            if v and isinstance(v, str) and len(v) >= 16:
                return v.strip()

    # 2) cookies: common cookie keys
    for ck in _MCP_COOKIE_KEYS:
        try:
            vv = resp.cookies.get(ck)
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


# ------------------ Response Parsing ------------------
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


# ------------------ Main MCP Call Function ------------------
def mcp_call(tool_name: str, arguments: Dict[str, Any], timeout: float = 10.0) -> Dict[str, Any]:
    """
    Call the MCP tool with immediate SSE response handling.
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


# ------------------ High-level Tool Wrappers ------------------
def sense_add(record: Dict) -> Dict:
    """
    Wrapper around mcp_call('sense_add', ...) with better diagnostics and timeout.
    """
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


# ------------------ Cleanup ------------------
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


# Register cleanup on exit
atexit.register(_close_mcp_client)