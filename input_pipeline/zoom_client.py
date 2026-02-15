"""
Zoom OAuth and API client.
Handles: authorize URL, token exchange, create meeting.
"""
import base64
import json
import logging
import os
import time
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

# Zoom OAuth endpoints
ZOOM_AUTH_URL = "https://zoom.us/oauth/authorize"
ZOOM_TOKEN_URL = "https://zoom.us/oauth/token"
ZOOM_API_BASE = "https://api.zoom.us/v2"

# Scopes — must match what's enabled in your Zoom app
ZOOM_SCOPES = "user:read:zak meeting:write:meeting"


def _token_file() -> Path:
    """Path to store OAuth tokens (dev only)."""
    return Path(__file__).parent.parent / ".zoom_tokens.json"


def _meeting_file() -> Path:
    """Path to store persistent meeting (reused for every call)."""
    return Path(__file__).parent.parent / ".zoom_meeting.json"


def _load_tokens() -> dict:
    """Load stored tokens from file."""
    f = _token_file()
    if f.exists():
        try:
            return json.loads(f.read_text())
        except Exception:
            pass
    return {}


def _save_tokens(tokens: dict) -> None:
    """Save tokens to file."""
    f = _token_file()
    try:
        f.write_text(json.dumps(tokens, indent=2))
    except Exception as e:
        logger.warning("Could not save Zoom tokens: %s", e)


def get_authorize_url() -> str:
    """
    Build the OAuth authorize URL for the user to visit.
    User will sign in to Zoom and be redirected back to our callback.
    """
    client_id = os.environ.get("ZOOM_CLIENT_ID", "")
    redirect_uri = os.environ.get("ZOOM_REDIRECT_URI", "http://localhost:3000/zoom/oauth/callback")
    if not client_id:
        raise ValueError("ZOOM_CLIENT_ID not set")
    params = {
        "response_type": "code",
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "scope": ZOOM_SCOPES,
    }
    qs = "&".join(f"{k}={v}" for k, v in params.items())
    return f"{ZOOM_AUTH_URL}?{qs}"


def exchange_code_for_tokens(code: str) -> dict:
    """
    Exchange authorization code for access_token and refresh_token.
    Called by the OAuth callback after user authorizes.
    """
    client_id = os.environ.get("ZOOM_CLIENT_ID", "")
    client_secret = os.environ.get("ZOOM_CLIENT_SECRET", "")
    redirect_uri = os.environ.get("ZOOM_REDIRECT_URI", "http://localhost:3000/zoom/oauth/callback")
    if not client_id or not client_secret:
        raise ValueError("ZOOM_CLIENT_ID and ZOOM_CLIENT_SECRET must be set")

    credentials = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
    headers = {"Authorization": f"Basic {credentials}", "Content-Type": "application/x-www-form-urlencoded"}
    data = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": redirect_uri,
    }

    with httpx.Client() as client:
        resp = client.post(ZOOM_TOKEN_URL, headers=headers, data=data)
        resp.raise_for_status()
        tokens = resp.json()

    _save_tokens_with_expiry(tokens)
    logger.info("Zoom OAuth tokens obtained and saved")
    return tokens


def _save_tokens_with_expiry(tokens: dict) -> None:
    """Save tokens and record expiry time (access_token expires in ~1 hour)."""
    expires_in = tokens.get("expires_in", 3600)
    tokens["_expires_at"] = time.time() + expires_in - 60  # refresh 1 min early
    _save_tokens(tokens)


def _refresh_tokens() -> dict:
    """Refresh access token using refresh_token."""
    tokens = _load_tokens()
    refresh = tokens.get("refresh_token")
    if not refresh:
        raise ValueError("No refresh token. Reconnect Zoom (click Connect Zoom).")

    client_id = os.environ.get("ZOOM_CLIENT_ID", "")
    client_secret = os.environ.get("ZOOM_CLIENT_SECRET", "")
    if not client_id or not client_secret:
        raise ValueError("ZOOM_CLIENT_ID and ZOOM_CLIENT_SECRET must be set")

    credentials = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
    headers = {"Authorization": f"Basic {credentials}", "Content-Type": "application/x-www-form-urlencoded"}
    data = {"grant_type": "refresh_token", "refresh_token": refresh}

    with httpx.Client() as client:
        resp = client.post(ZOOM_TOKEN_URL, headers=headers, data=data)
        resp.raise_for_status()
        new_tokens = resp.json()

    _save_tokens_with_expiry(new_tokens)
    logger.info("Zoom access token refreshed")
    return new_tokens


def _get_access_token() -> str:
    """Get current access token, refreshing if expired."""
    tokens = _load_tokens()
    access = tokens.get("access_token")
    if not access:
        raise ValueError("Not authenticated with Zoom. Connect Zoom first.")

    expires_at = tokens.get("_expires_at", 0)
    if time.time() >= expires_at:
        tokens = _refresh_tokens()
        access = tokens.get("access_token")
        if not access:
            raise ValueError("Token refresh failed. Reconnect Zoom.")

    return access


def _load_persistent_meeting():
    """Load stored meeting if it exists."""
    f = _meeting_file()
    if f.exists():
        try:
            return json.loads(f.read_text())
        except Exception:
            pass
    return None


def _save_persistent_meeting(meeting: dict) -> None:
    """Store meeting for reuse."""
    f = _meeting_file()
    try:
        f.write_text(json.dumps(meeting, indent=2))
    except Exception as e:
        logger.warning("Could not save persistent meeting: %s", e)


def create_meeting(topic: str = "Learning Companion Call", duration_minutes: int = 60) -> dict:
    """
    Create a Zoom meeting via REST API.
    Returns meeting details including join_url, id, password.
    """
    token = _get_access_token()
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    payload = {
        "topic": topic,
        "type": 1,  # Instant meeting
        "duration": duration_minutes,
        "settings": {
            "join_before_host": True,
            "mute_upon_entry": False,
            "watermark": False,
        },
    }

    with httpx.Client() as client:
        resp = client.post(f"{ZOOM_API_BASE}/users/me/meetings", headers=headers, json=payload)
        resp.raise_for_status()
        return resp.json()


def reset_persistent_meeting() -> None:
    """Delete stored meeting so the next call creates a new one."""
    f = _meeting_file()
    if f.exists():
        try:
            f.unlink()
            logger.info("Persistent meeting reset")
        except Exception as e:
            logger.warning("Could not reset meeting: %s", e)


def get_or_create_persistent_meeting(topic: str = "Learning Companion Call") -> dict:
    """
    Return the same meeting every time. Creates once, stores in .zoom_meeting.json, reuses.
    Share the join_url with others or join from another account first.
    """
    stored = _load_persistent_meeting()
    if stored and stored.get("meeting_id") and stored.get("join_url"):
        return stored

    meeting = create_meeting(topic=topic)
    meeting_id = meeting.get("id", "")
    password = meeting.get("password", "")
    join_url = f"https://zoom.us/wc/join/{meeting_id}" + (f"?pwd={password}" if password else "")
    stored = {
        "meeting_id": str(meeting_id),
        "join_url": join_url,
        "password": password or "",
    }
    _save_persistent_meeting(stored)
    logger.info("Created and saved persistent meeting %s", meeting_id)
    return stored


def is_connected() -> bool:
    """Check if we have valid Zoom tokens."""
    tokens = _load_tokens()
    return bool(tokens.get("access_token"))
