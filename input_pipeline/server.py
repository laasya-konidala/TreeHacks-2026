"""
FastAPI server — context ingestion, WebSocket broadcast, reply forwarding.
Merges data from two sources:
  1. Electron/Gemini VLM (screen analysis: topic, stuck, work_status, confusion)
  2. Chrome extension (behavioral: typing speed, deletions, pauses, scroll-back)
Runs on port 8080.
"""
import asyncio
import json
import logging
import time
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logger = logging.getLogger(__name__)

app = FastAPI(title="Ambient Learning Agent Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── In-memory state ───
# Stores the latest MERGED context (Gemini VLM + behavioral signals)
latest_context: dict = {"data": None, "timestamp": 0}

# Separate buffers for the two data sources so we can merge them
_gemini_buffer: dict = {"data": None, "timestamp": 0}
_behavioral_buffer: dict = {"data": None, "timestamp": 0}

ws_clients: list[WebSocket] = []
reply_queue: asyncio.Queue = asyncio.Queue()


# ─── Models ───
class TouchRequest(BaseModel):
    message: str = ""
    user_id: str = "default"


# ─── Context Merging ───

def _merge_context() -> dict:
    """
    Merge Gemini VLM analysis with Chrome extension behavioral signals
    into a single WorkContext dict. Gemini provides the 'eyes' (topic,
    stuck, work_status), Chrome provides the 'fingers' (typing, deletions,
    pauses, scrolling).
    """
    gemini = _gemini_buffer["data"] or {}
    behavioral = _behavioral_buffer["data"] or {}

    merged = {
        # Gemini VLM provides these (it actually sees the screen)
        "screen_content": gemini.get("screen_content") or behavioral.get("screen_content", ""),
        "screen_content_type": gemini.get("screen_content_type") or behavioral.get("screen_content_type", "text"),
        "detected_topic": gemini.get("detected_topic") or behavioral.get("detected_topic", ""),
        "detected_subtopic": gemini.get("detected_subtopic") or behavioral.get("detected_subtopic", ""),

        # Chrome extension provides these (it monitors keystrokes)
        "typing_speed_ratio": behavioral.get("typing_speed_ratio", gemini.get("typing_speed_ratio", 1.0)),
        "deletion_rate": behavioral.get("deletion_rate", gemini.get("deletion_rate", 0.0)),
        "pause_duration": behavioral.get("pause_duration", gemini.get("pause_duration", 0.0)),
        "scroll_back_count": behavioral.get("scroll_back_count", gemini.get("scroll_back_count", 0)),

        # Verbal cues can come from either source
        "audio_transcript": gemini.get("audio_transcript") or behavioral.get("audio_transcript"),
        "verbal_confusion_cues": (
            gemini.get("verbal_confusion_cues", []) +
            behavioral.get("verbal_confusion_cues", [])
        ),

        # Touch / user message — either source
        "user_touched_agent": (
            behavioral.get("user_touched_agent", False) or
            gemini.get("user_touched_agent", False)
        ),
        "user_message": behavioral.get("user_message") or gemini.get("user_message"),

        # No screenshot needed (Gemini already analyzed the screen)
        "screenshot_b64": None,

        # IDs
        "user_id": behavioral.get("user_id") or gemini.get("user_id", "default"),
        "session_id": gemini.get("session_id") or behavioral.get("session_id", ""),
        "timestamp": gemini.get("timestamp") or behavioral.get("timestamp", ""),

        # Gemini VLM analysis fields
        "gemini_stuck": gemini.get("gemini_stuck", False),
        "gemini_work_status": gemini.get("gemini_work_status", "unclear"),
        "gemini_confused_about": gemini.get("gemini_confused_about", []),
        "gemini_understands": gemini.get("gemini_understands", []),
        "gemini_error": gemini.get("gemini_error"),
        "gemini_mode": gemini.get("gemini_mode", ""),
        "gemini_notes": gemini.get("gemini_notes", ""),
    }

    return merged


# ─── Endpoints ───

@app.post("/context")
async def receive_context(ctx: dict):
    """
    Receive context from either Gemini (Electron) or Chrome extension.
    Automatically detects the source and updates the right buffer,
    then merges into the unified context that the orchestrator polls.
    """
    source = ctx.get("_source", "")

    if source == "chrome_extension":
        # Behavioral signals from Chrome extension
        _behavioral_buffer["data"] = ctx
        _behavioral_buffer["timestamp"] = time.time()
    else:
        # Gemini VLM analysis from Electron (or direct POST)
        _gemini_buffer["data"] = ctx
        _gemini_buffer["timestamp"] = time.time()

    # Merge both sources into the unified context
    latest_context["data"] = _merge_context()
    latest_context["timestamp"] = time.time()

    return {"status": "ok", "source": source or "gemini"}


@app.get("/context/latest")
async def get_latest():
    """
    Get the latest merged WorkContext (consumed on read).
    Orchestrator polls this every 2 seconds.
    """
    data = latest_context["data"]
    if data:
        latest_context["data"] = None
        return data
    return {}


@app.post("/reply")
async def receive_reply(reply: dict):
    """
    Receive user reply from Electron overlay.
    Forwards to orchestrator via reply queue.
    """
    await reply_queue.put(reply)
    return {"status": "ok"}


@app.get("/reply/poll")
async def poll_reply():
    """Poll for user replies (used by orchestrator)."""
    try:
        reply = reply_queue.get_nowait()
        return reply
    except asyncio.QueueEmpty:
        return {}


@app.post("/touch")
async def explicit_touch(body: TouchRequest):
    """Handle explicit help request (Ctrl+Shift+H from Chrome or Electron)."""
    # Set touch flag on both buffers so the merged context picks it up
    if _behavioral_buffer["data"]:
        _behavioral_buffer["data"]["user_touched_agent"] = True
    if _gemini_buffer["data"]:
        _gemini_buffer["data"]["user_touched_agent"] = True

    # Also set user message
    for buf in [_behavioral_buffer, _gemini_buffer]:
        if buf["data"]:
            buf["data"]["user_message"] = body.message

    # Re-merge
    latest_context["data"] = _merge_context()
    latest_context["timestamp"] = time.time()

    return {"status": "ok"}


@app.post("/agent-response")
async def agent_response(response: dict):
    """
    Receive agent response and broadcast to all WebSocket clients.
    Called by orchestrator when a specialist agent responds.
    """
    disconnected = []
    for ws in ws_clients:
        try:
            await ws.send_json(response)
        except Exception:
            disconnected.append(ws)

    for ws in disconnected:
        try:
            ws_clients.remove(ws)
        except ValueError:
            pass

    return {"status": "ok", "broadcast_count": len(ws_clients)}


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    """WebSocket endpoint for Electron overlay to receive agent responses."""
    await ws.accept()
    ws_clients.append(ws)
    logger.info(f"WebSocket client connected. Total: {len(ws_clients)}")

    try:
        while True:
            # Keep alive — client may send pings or messages
            data = await ws.receive_text()
            # Could handle client messages here if needed
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        try:
            ws_clients.remove(ws)
        except ValueError:
            pass
        logger.info(f"WebSocket client disconnected. Total: {len(ws_clients)}")


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "ws_clients": len(ws_clients),
        "has_context": latest_context["data"] is not None,
        "has_gemini_data": _gemini_buffer["data"] is not None,
        "has_behavioral_data": _behavioral_buffer["data"] is not None,
    }
