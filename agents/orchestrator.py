"""
Orchestrator Agent — Routes to the right learning agent based on VLM context.

Decision flow:
  1. VLM says what student is doing → pick agent (conceptual / applied / extension)
  2. BKT says mastery level → calibrate exercise difficulty
  3. Timing logic detects natural moments → decide WHEN to prompt
"""
import asyncio
import json
import logging
import time
from typing import Optional

import httpx
from uagents import Agent, Context

from agents.config import (
    ORCHESTRATOR_SEED, ORCHESTRATOR_PORT, BACKEND_URL,
    GEMINI_API_KEY, GEMINI_MODEL,
    AGENTVERSE_ENABLED, AGENTVERSE_URL,
)
from agents.models import VLMContext, AgentRequest, AgentResponse, TimingSignal
from agents.learner_model import ConfidenceWeightedBKT

logger = logging.getLogger(__name__)

# ─── Agent Setup ───
orchestrator = Agent(
    name="learning_orchestrator",
    port=ORCHESTRATOR_PORT,
    seed=ORCHESTRATOR_SEED,
    endpoint=[f"http://127.0.0.1:{ORCHESTRATOR_PORT}/submit"],
    agentverse=AGENTVERSE_URL if AGENTVERSE_ENABLED else None,
    mailbox=AGENTVERSE_ENABLED,
    description=(
        "Learning orchestrator — observes VLM screen analysis, detects natural "
        "moments to prompt, and routes to the right learning agent."
    ),
)

# ─── State ───
bkt = ConfidenceWeightedBKT()
state = {
    "last_prompt_time": 0.0,
    "last_topic": "",
    "last_mode": "",
    "same_content_since": 0.0,    # when we first saw this topic
    "prompt_count": 0,
    "stuck_count": 0,              # consecutive incomplete/incorrect observations
    "observations": [],            # rolling buffer of VLM observations
    "agent_addresses": {
        "conceptual": None,
        "applied": None,
        "extension": None,
    },
}

# ─── Config ───
MIN_SECONDS_BETWEEN_PROMPTS = 45    # hard cooldown — never prompt faster than this
NATURAL_PAUSE_MIN_SECONDS = 25      # need at least 25s on topic before natural_pause triggers
STUCK_THRESHOLD_SECONDS = 60        # 60s on same topic + VLM says stuck → stuck trigger
FALLBACK_PROMPT_SECONDS = 180       # 3 min — safety net so system doesn't go completely silent
MAX_OBSERVATIONS = 20
STUCK_OBSERVATION_COUNT = 5         # how many "incomplete" work_status in a row = stuck


def _resolve_agent_addresses():
    """Resolve agent addresses from seeds."""
    from uagents import Agent as _Agent

    seeds = {
        "conceptual": "ambient_learning_conceptual_seed_2026",
        # "applied": "ambient_learning_applied_seed_2026",     # TODO: add later
        # "extension": "ambient_learning_extension_seed_2026",  # TODO: add later
    }

    for name, seed in seeds.items():
        tmp = _Agent(name=f"_tmp_{name}", seed=seed)
        state["agent_addresses"][name] = str(tmp.address)
        logger.info(f"[Orchestrator] {name} address: {tmp.address}")


# ─── Timing: Should we prompt now? ──────────────────────────────────
def should_prompt_now(vlm: VLMContext) -> tuple[bool, str]:
    """
    Decide if NOW is a good time to prompt the student.
    Returns (should_prompt, reason).

    Priority chain:
      1. Hard cooldown (30s) — always respected
      2. Immediate triggers: topic_transition, mode_change
      3. Natural pause: VLM says natural_pause AND 20s+ on topic
      4. Stuck: VLM says stuck AND 60s+ on topic (or repeated incomplete work)
      5. Fallback: 90s on same topic — safety net so system doesn't go silent
    """
    now = time.time()

    # ── 1. Hard cooldown — never prompt faster than this ──
    time_since_last = now - state["last_prompt_time"]
    if time_since_last < MIN_SECONDS_BETWEEN_PROMPTS:
        return False, "cooldown"

    # ── Track topic duration ──
    if vlm.topic != state["last_topic"]:
        old_topic = state["last_topic"]
        state["same_content_since"] = now
        state["last_topic"] = vlm.topic
        state["stuck_count"] = 0  # reset stuck counter on topic change

        # ── 2a. Topic transition — good moment to prompt ──
        if old_topic:
            return True, "topic_transition"

    seconds_on_topic = now - state["same_content_since"]

    # ── 2b. Mode changed (CONCEPTUAL → APPLIED, etc.) ──
    if vlm.mode and vlm.mode != state["last_mode"] and state["last_mode"]:
        state["last_mode"] = vlm.mode
        return True, "mode_change"
    state["last_mode"] = vlm.mode or state["last_mode"]

    # ── 3. Natural pause: VLM detected a pause AND enough time on topic ──
    # VLM sees: video paused, finished writing, scrolled to new section
    vlm_says_pause = getattr(vlm, 'notes', '') and 'pause' in getattr(vlm, 'notes', '').lower()
    # Also check the raw data for the natural_pause field from the JSON
    raw_data = {}
    try:
        import json as _json
        raw_data = _json.loads(vlm.raw_vlm_text) if vlm.raw_vlm_text else {}
    except Exception:
        pass
    natural_pause_detected = raw_data.get("gemini_natural_pause", False) or vlm_says_pause

    if natural_pause_detected and seconds_on_topic >= NATURAL_PAUSE_MIN_SECONDS:
        return True, "natural_pause"

    # ── 4. Stuck: sustained lack of progress ──
    # Track consecutive "incomplete" or "incorrect" work_status observations
    if vlm.work_status in ("incomplete", "incorrect"):
        state["stuck_count"] = state.get("stuck_count", 0) + 1
    elif vlm.work_status == "correct":
        state["stuck_count"] = 0  # they made progress

    stuck_by_vlm = vlm.stuck and seconds_on_topic >= STUCK_THRESHOLD_SECONDS
    stuck_by_history = state.get("stuck_count", 0) >= STUCK_OBSERVATION_COUNT

    if stuck_by_vlm or stuck_by_history:
        return True, "stuck"

    # ── 5. Fallback — don't go silent forever ──
    if seconds_on_topic >= FALLBACK_PROMPT_SECONDS:
        return True, "fallback"

    return False, "not_yet"


# ─── Route: Which agent should handle this? ─────────────────────────
def pick_agent(vlm: VLMContext) -> str:
    """
    Pick which agent based on what the student is DOING.
    For now, only conceptual exists — others return "conceptual" as fallback.
    """
    mode = (vlm.mode or "").upper()

    if mode == "APPLIED":
        # TODO: route to applied agent when it exists
        return "conceptual"  # fallback for now
    elif mode == "CONSOLIDATION":
        # TODO: route to extension agent when it exists
        return "conceptual"  # fallback for now
    else:
        return "conceptual"


# ─── Poll for new context from the FastAPI backend ──────────────────
@orchestrator.on_interval(period=8.0)
async def poll_context(ctx: Context):
    """Poll the FastAPI backend for the latest VLM screen analysis."""
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            resp = await client.get(f"{BACKEND_URL}/context/latest")
            if resp.status_code != 200:
                return

            data = resp.json()
            if not data or not data.get("detected_topic"):
                return

            # Build VLMContext from the backend data
            vlm = VLMContext(
                activity=data.get("screen_content", ""),
                topic=data.get("detected_topic", ""),
                subtopic=data.get("detected_subtopic", ""),
                mode=data.get("gemini_mode", ""),
                content_type=data.get("screen_content_type", "text"),
                work_status=data.get("gemini_work_status", "unclear"),
                stuck=data.get("gemini_stuck", False),
                error_description=data.get("gemini_error"),
                notes=data.get("gemini_notes", ""),
                speech_transcript=data.get("audio_transcript"),
                raw_vlm_text=json.dumps(data, default=str),
            )

            # Add to observation buffer
            obs_summary = f"{vlm.activity} — {vlm.topic} ({vlm.mode})"
            state["observations"].append(obs_summary)
            if len(state["observations"]) > MAX_OBSERVATIONS:
                state["observations"] = state["observations"][-MAX_OBSERVATIONS:]

            # Update BKT if we have topic info
            if vlm.topic:
                bkt.init_concept(vlm.topic)

                # Use work_status as a signal for BKT
                if vlm.work_status == "correct":
                    bkt.update(vlm.topic, correct=True, confidence=0.7, source="screen")
                elif vlm.work_status == "incorrect":
                    bkt.update(vlm.topic, correct=False, confidence=0.7, source="screen")

            # Should we prompt now?
            should, reason = should_prompt_now(vlm)
            if not should:
                return

            logger.info(f"[Orchestrator] Prompting — reason: {reason}, topic: {vlm.topic}")

            # Pick the agent
            agent_name = pick_agent(vlm)
            agent_addr = state["agent_addresses"].get(agent_name)
            if not agent_addr:
                logger.warning(f"[Orchestrator] No address for {agent_name}")
                return

            # Build the request
            mastery = bkt.get_mastery(vlm.topic) if vlm.topic else 0.0
            quality = bkt.get_observation_quality(vlm.topic) if vlm.topic else {}

            request = AgentRequest(
                vlm_context=vlm,
                mastery=mastery,
                mastery_quality=quality.get("quality", "no_data"),
                trigger_reason=reason,
                recent_observations=state["observations"][-5:],
                session_id=f"session_{int(time.time())}",
            )

            # Send to the agent
            await ctx.send(agent_addr, request)
            state["last_prompt_time"] = time.time()
            state["prompt_count"] += 1
            state["same_content_since"] = time.time()  # reset timer

            logger.info(f"[Orchestrator] Sent to {agent_name} (prompt #{state['prompt_count']})")

    except httpx.ConnectError:
        pass  # backend not running yet, that's fine
    except Exception as e:
        logger.error(f"[Orchestrator] Error: {e}")


# ─── Handle responses from agents ───────────────────────────────────
@orchestrator.on_message(model=AgentResponse)
async def handle_agent_response(ctx: Context, sender: str, msg: AgentResponse):
    """Forward agent response to the sidebar via the FastAPI backend."""
    logger.info(f"[Orchestrator] Got response from {msg.agent_type}: {msg.content[:80]}")

    # Forward to the FastAPI backend which sends it to the sidebar via WebSocket
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            await client.post(
                f"{BACKEND_URL}/agent-response",
                json={
                    "agent_type": msg.agent_type,
                    "content": msg.content,
                    "tool_used": msg.tool_used,
                    "topic": msg.topic,
                    "mastery": msg.mastery,
                },
            )
    except Exception as e:
        logger.error(f"[Orchestrator] Failed to forward response: {e}")


# ─── Startup ───
@orchestrator.on_event("startup")
async def on_startup(ctx: Context):
    """Resolve agent addresses on startup."""
    _resolve_agent_addresses()
    logger.info("[Orchestrator] Ready — polling for VLM context every 3s")
    logger.info(f"[Orchestrator] Min seconds between prompts: {MIN_SECONDS_BETWEEN_PROMPTS}")
