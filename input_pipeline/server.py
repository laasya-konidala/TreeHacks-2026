"""
FastAPI server — context ingestion, WebSocket broadcast, reply forwarding.
Merges data from two sources:
  1. Electron/Gemini VLM (screen analysis: topic, stuck, work_status, confusion)
  2. Chrome extension (behavioral: typing speed, deletions, pauses, scroll-back)
Runs on port 8080.
Also: Zoom OAuth and meeting creation.
"""
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import asyncio
import json
import logging
import os
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import httpx
from input_pipeline.zoom_client import (
    exchange_code_for_tokens,
    get_authorize_url,
    get_or_create_persistent_meeting,
    is_connected,
    reset_persistent_meeting,
)

logger = logging.getLogger(__name__)

app = FastAPI(title="Ambient Learning Agent Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Manim output directory & static serving ───
MANIM_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "manim_output"
MANIM_OUTPUT_DIR.mkdir(exist_ok=True)
app.mount("/video", StaticFiles(directory=str(MANIM_OUTPUT_DIR)), name="manim_videos")

# Track render jobs: job_id → { status, url, error }
_manim_jobs: dict[str, dict] = {}

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


class CreateMeetingRequest(BaseModel):
    topic: str = "Learning Companion Call"


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
        "gemini_screen_details": gemini.get("gemini_screen_details", ""),
        "gemini_natural_pause": gemini.get("gemini_natural_pause", False),
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

    topic = latest_context["data"].get("detected_topic", "?")
    mode = latest_context["data"].get("gemini_mode", "?")
    logger.info(f"[Context] Received from {source or 'gemini'} — topic: {topic}, mode: {mode}")

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


# ─── Manim rendering ───

class ManimRenderRequest(BaseModel):
    code: str
    session_id: str = ""


@app.post("/manim/render")
async def manim_render(req: ManimRenderRequest):
    """
    Accept a Manim script, render it in a background thread, and return a job_id.
    The overlay polls /manim/status/{job_id} until the video is ready.
    """
    job_id = uuid.uuid4().hex[:12]
    _manim_jobs[job_id] = {"status": "rendering", "url": None, "error": None}

    asyncio.get_event_loop().run_in_executor(None, _run_manim_render, job_id, req.code)

    return {"job_id": job_id, "status_url": f"/manim/status/{job_id}"}


@app.get("/manim/status/{job_id}")
async def manim_status(job_id: str):
    """Poll render status. Returns {status, url, error}."""
    job = _manim_jobs.get(job_id)
    if not job:
        return {"status": "error", "error": "Unknown job_id"}
    return job


def _run_manim_render(job_id: str, code: str):
    """
    Run manim CLI in a subprocess. Writes the script to a temp file,
    renders to mp4 in manim_output/, and updates _manim_jobs.
    """
    try:
        # Write script to a temp file
        script_path = MANIM_OUTPUT_DIR / f"{job_id}.py"
        script_path.write_text(code, encoding="utf-8")

        # Find the Scene class name from the code
        import re
        scene_match = re.search(r"class\s+(\w+)\s*\(.*Scene.*\)", code)
        scene_name = scene_match.group(1) if scene_match else "ConceptScene"

        # Run manim CLI: render at 720p, output to manim_output/
        result = subprocess.run(
            [
                "manim", "render",
                "-ql",  # low quality (480p) for speed; use -qm for 720p
                "--format", "mp4",
                "--media_dir", str(MANIM_OUTPUT_DIR / "media"),
                str(script_path),
                scene_name,
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )

        if result.returncode != 0:
            logger.error(f"[Manim] Render failed for {job_id}: {result.stderr[-500:]}")
            _manim_jobs[job_id] = {
                "status": "error",
                "url": None,
                "error": result.stderr[-300:] or "Render failed",
            }
            return

        # Find the rendered mp4 in the media directory
        media_dir = MANIM_OUTPUT_DIR / "media" / "videos" / f"{job_id}" / "480p15"
        mp4_files = list(media_dir.glob("*.mp4")) if media_dir.exists() else []

        if not mp4_files:
            # Also check other quality directories
            for quality_dir in (MANIM_OUTPUT_DIR / "media" / "videos" / f"{job_id}").glob("*"):
                mp4_files = list(quality_dir.glob("*.mp4"))
                if mp4_files:
                    break

        if not mp4_files:
            _manim_jobs[job_id] = {
                "status": "error",
                "url": None,
                "error": "Render succeeded but no mp4 found",
            }
            return

        # Move the mp4 to the root manim_output/ for simple static serving
        final_name = f"{job_id}.mp4"
        final_path = MANIM_OUTPUT_DIR / final_name
        mp4_files[0].rename(final_path)

        _manim_jobs[job_id] = {
            "status": "ready",
            "url": f"/video/{final_name}",
            "error": None,
        }
        logger.info(f"[Manim] Render complete: {final_name}")

    except subprocess.TimeoutExpired:
        _manim_jobs[job_id] = {
            "status": "error",
            "url": None,
            "error": "Render timed out (120s limit)",
        }
    except Exception as e:
        logger.error(f"[Manim] Unexpected error for {job_id}: {e}")
        _manim_jobs[job_id] = {
            "status": "error",
            "url": None,
            "error": str(e)[:300],
        }


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


# ─── Zoom OAuth and Meetings ───

@app.get("/zoom/auth")
async def zoom_auth_url():
    """Get Zoom OAuth authorize URL. Open this in browser to connect Zoom."""
    try:
        url = get_authorize_url()
        return {"url": url}
    except ValueError as e:
        return {"error": str(e), "url": None}


@app.get("/zoom/oauth/callback")
async def zoom_oauth_callback(code: str = Query(...)):
    """
    OAuth callback. Zoom redirects here after user authorizes.
    Exchange code for tokens, then redirect to a success page.
    """
    try:
        exchange_code_for_tokens(code)
        # Redirect to a simple success page (we'll host this or use data URL)
        return RedirectResponse(url="data:text/html,<h1>Zoom connected!</h1><p>You can close this tab and return to the app.</p>")
    except Exception as e:
        logger.exception("Zoom OAuth callback failed")
        return RedirectResponse(
            url=f"data:text/html,<h1>Error</h1><p>{str(e)}</p><p>Check server logs.</p>"
        )


@app.get("/realtime/config")
async def realtime_config():
    """
    Get ephemeral key and instructions for OpenAI Realtime voice agent.
    Requires OPENAI_API_KEY and REALTIME_INSTRUCTIONS in .env.
    """
    import os
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    instructions = os.environ.get("REALTIME_INSTRUCTIONS", "You are a helpful assistant.").strip()
    if not api_key or api_key.startswith("paste_"):
        return {"error": "OPENAI_API_KEY not set. Add it to .env"}
    try:
        session_config = {
            "session": {
                "type": "realtime",
                "model": "gpt-realtime",
                "instructions": instructions,
                "audio": {"output": {"voice": "marin"}},
            }
        }
        async with httpx.AsyncClient() as client:
            r = await client.post(
                "https://api.openai.com/v1/realtime/client_secrets",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json=session_config,
                timeout=15.0,
            )
            r.raise_for_status()
            data = r.json()
        ephemeral_key = data.get("value") or data.get("client_secret", {}).get("value")
        if not ephemeral_key:
            return {"error": "No ephemeral key in response"}
        return {"apiKey": ephemeral_key, "instructions": instructions}
    except httpx.HTTPStatusError as e:
        try:
            err = e.response.json()
            msg = err.get("error", {}).get("message", str(e))
        except Exception:
            msg = str(e)
        return {"error": f"OpenAI API error: {msg}"}


@app.get("/zoom/status")
async def zoom_status():
    """Check if Zoom is connected (has valid tokens)."""
    return {"connected": is_connected()}


@app.post("/zoom/reset-meeting")
async def zoom_reset_meeting():
    """Clear stored meeting so the next call creates a new one."""
    reset_persistent_meeting()
    return {"ok": True}


@app.post("/zoom/create-meeting")
async def zoom_create_meeting(body: Optional[CreateMeetingRequest] = None):
    """
    Create a Zoom meeting. Returns join_url, meeting_id, etc.
    Requires Zoom to be connected first (OAuth).
    """
    topic = (body or CreateMeetingRequest()).topic
    try:
        meeting = get_or_create_persistent_meeting(topic=topic)
        return {
            "join_url": meeting["join_url"],
            "meeting_id": meeting["meeting_id"],
            "password": meeting.get("password", ""),
        }
    except ValueError as e:
        return {"error": str(e), "join_url": None}
    except httpx.HTTPStatusError as e:
        try:
            err_body = e.response.json()
            msg = err_body.get("message", err_body.get("reason", str(e)))
        except Exception:
            msg = str(e)
        logger.warning("Zoom API error %s: %s", e.response.status_code, msg)
        return {"error": f"Zoom API error: {msg}", "join_url": None}
