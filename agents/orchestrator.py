"""
Orchestrator Agent — Routes to the right learning agent based on VLM context.

Also the public-facing agent for ASI:One: includes the Chat Protocol
(for discoverability) and Payment Protocol (for monetization).

Decision flow:
  1. VLM says what student is doing → pick agent (conceptual / applied / extension)
  2. BKT says mastery level → calibrate exercise difficulty
  3. Timing logic detects natural moments → decide WHEN to prompt

ASI:One flow:
  - User sends ChatMessage via ASI:One → chat_proto handles it
  - User can pay via Payment Protocol → payment_proto handles it
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
    CONCEPTUAL_SEED, APPLIED_SEED, EXTENSION_SEED,
    GEMINI_API_KEY, GEMINI_MODEL,
    AGENTVERSE_ENABLED, AGENTVERSE_URL,
)
from agents.models import VLMContext, AgentRequest, AgentResponse, TimingSignal
from agents.learner_model import ConfidenceWeightedBKT

# Import ASI:One compatible protocols
from agents.chat_protocol import chat_proto
from agents.payment_protocol import payment_proto, tier_protocol, set_agent_wallet

logger = logging.getLogger(__name__)

# ─── Agent Setup ───
_orch_kwargs = dict(
    name="learning_orchestrator",
    port=ORCHESTRATOR_PORT,
    seed=ORCHESTRATOR_SEED,
    description=(
        "Ambient Learning Orchestrator — an AI tutoring agent that observes what "
        "you're studying (via screen analysis), detects when you need help, and "
        "provides contextual questions, visualizations, and guided problem-solving. "
        "Ask me about any topic — math, science, programming, or anything you're "
        "learning. I can explain concepts, help you work through problems, and "
        "push you to make connections across topics."
    ),
)
if AGENTVERSE_ENABLED:
    _orch_kwargs["mailbox"] = True
    _orch_kwargs["publish_agent_details"] = True
else:
    _orch_kwargs["endpoint"] = [f"http://127.0.0.1:{ORCHESTRATOR_PORT}/submit"]

orchestrator = Agent(**_orch_kwargs)

# ─── Include ASI:One Protocols (publish_manifest=True makes them discoverable) ───
orchestrator.include(chat_proto, publish_manifest=True)
orchestrator.include(payment_proto, publish_manifest=True)
orchestrator.include(tier_protocol, publish_manifest=True)

# ─── State ───
bkt = ConfidenceWeightedBKT()
state = {
    "last_prompt_time": 0.0,
    "last_topic": "",
    "last_mode": "",
    "same_content_since": 0.0,    # when we first saw this topic
    "prompt_count": 0,
    "observations": [],            # rolling buffer of VLM observations
    "agent_addresses": {
        "conceptual": None,
        "applied": None,
        "extension": None,
    },
}

# ─── Config ───
MIN_SECONDS_BETWEEN_PROMPTS = 30   # don't spam
NATURAL_PAUSE_THRESHOLD = 15       # seconds on same content before considering a prompt
MAX_OBSERVATIONS = 20


def _resolve_agent_addresses():
    """Resolve agent addresses from seeds."""
    from uagents import Agent as _Agent

    seeds = {
        "conceptual": CONCEPTUAL_SEED,
        "applied": APPLIED_SEED,
        "extension": EXTENSION_SEED,
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
    """
    now = time.time()

    # Cooldown: don't prompt too often
    time_since_last = now - state["last_prompt_time"]
    if time_since_last < MIN_SECONDS_BETWEEN_PROMPTS:
        return False, "cooldown"

    # Track how long they've been on the same topic
    if vlm.topic != state["last_topic"]:
        # Topic changed — this is a natural transition
        state["same_content_since"] = now
        state["last_topic"] = vlm.topic

        # If we had a previous topic, this transition is a good moment
        if state["last_topic"]:
            return True, "topic_transition"

    seconds_on_topic = now - state["same_content_since"]

    # Natural pause: been on same content for a while
    if seconds_on_topic > NATURAL_PAUSE_THRESHOLD:
        return True, "natural_pause"

    # Student seems stuck (VLM detected)
    if vlm.stuck:
        return True, "stuck"

    # Student said something (speech)
    if vlm.speech_transcript:
        return True, "speech"

    # Mode changed (e.g., went from reading to solving)
    if vlm.mode and vlm.mode != state["last_mode"] and state["last_mode"]:
        state["last_mode"] = vlm.mode
        return True, "mode_change"

    state["last_mode"] = vlm.mode or state["last_mode"]
    return False, "not_yet"


# ─── Route: Which agent should handle this? ─────────────────────────
def pick_agent(vlm: VLMContext) -> str:
    """
    Pick which agent based on what the student is DOING.
    Now routes to all three agents.
    """
    mode = (vlm.mode or "").upper()

    if mode == "APPLIED":
        return "applied"
    elif mode == "CONSOLIDATION":
        return "extension"
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
    """Resolve agent addresses and set up wallet on startup."""
    _resolve_agent_addresses()

    # Inject wallet into payment protocol for on-chain verification
    set_agent_wallet(orchestrator.wallet)

    logger.info("[Orchestrator] Ready — polling for VLM context every 8s")
    logger.info(f"[Orchestrator] Agent address: {orchestrator.address}")
    logger.info(f"[Orchestrator] Chat Protocol: included (ASI:One discoverable)")
    logger.info(f"[Orchestrator] Payment Protocol: included (FET monetization)")
    logger.info(f"[Orchestrator] Agentverse: {'ENABLED' if AGENTVERSE_ENABLED else 'disabled'}")
    logger.info(f"[Orchestrator] Min seconds between prompts: {MIN_SECONDS_BETWEEN_PROMPTS}")

    for name, addr in state["agent_addresses"].items():
        logger.info(f"[Orchestrator] {name}: {addr}")
