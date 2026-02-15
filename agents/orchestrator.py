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
    "stuck_count": 0,              # consecutive incomplete/incorrect observations
    "observations": [],            # rolling buffer of VLM observations
    "agent_addresses": {
        "conceptual": None,
        "applied": None,
        "extension": None,
    },
}

# ─── Config ───
# ⚠️ TESTING VALUES — raise these back for production
MIN_SECONDS_BETWEEN_PROMPTS = 15    # hard cooldown (prod: 45)
NATURAL_PAUSE_MIN_SECONDS = 10      # time on topic before pause triggers (prod: 25)
STUCK_THRESHOLD_SECONDS = 30        # stuck timer (prod: 60)
FALLBACK_PROMPT_SECONDS = 40        # safety net (prod: 180)
MAX_OBSERVATIONS = 20
STUCK_OBSERVATION_COUNT = 5         # how many "incomplete" work_status in a row = stuck
_poll_count = 0                     # debug counter for VLM observations


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

    Priority chain:
      1. Hard cooldown (45s) — always respected
      2. Immediate triggers: topic_transition, mode_change
      3. Natural pause: VLM says natural_pause AND 25s+ on topic
      4. Stuck: VLM says stuck AND 60s+ on topic (or repeated incomplete work)
      5. Fallback: 3 min — safety net so system doesn't go silent
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
    vlm_says_pause = getattr(vlm, 'notes', '') and 'pause' in getattr(vlm, 'notes', '').lower()
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
    if vlm.work_status in ("incomplete", "incorrect"):
        state["stuck_count"] = state.get("stuck_count", 0) + 1
    elif vlm.work_status == "correct":
        state["stuck_count"] = 0

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
    Mode comes from the VLM's analysis of the screen.
    """
    mode = (vlm.mode or "").upper()

    if mode == "APPLIED":
        return "applied"
    elif mode == "CONSOLIDATION":
        return "extension"
    else:
        # CONCEPTUAL is the default — reading, watching, learning
        return "conceptual"


# ─── Poll for new context from the FastAPI backend ──────────────────
@orchestrator.on_interval(period=8.0)
async def poll_context(ctx: Context):
    """Poll the FastAPI backend for the latest VLM screen analysis."""
    global _poll_count
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            resp = await client.get(f"{BACKEND_URL}/context/latest")
            if resp.status_code != 200:
                return

            data = resp.json()
            if not data:
                return  # empty — no VLM data posted yet
            if not data.get("detected_topic"):
                logger.info(f"  [poll] Got data but no detected_topic. Keys: {list(data.keys())[:8]}")
                return

            _poll_count += 1

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

            # ─── DEBUG: VLM observation (detailed every 3rd, compact otherwise) ───
            now = time.time()
            secs_on_topic = now - state["same_content_since"] if state["same_content_since"] else 0
            cooldown_left = max(0, MIN_SECONDS_BETWEEN_PROMPTS - (now - state["last_prompt_time"]))
            screen_details = data.get("gemini_screen_details", "")[:100]

            if _poll_count % 3 == 0:
                logger.info(
                    f"\n{'─' * 60}\n"
                    f"  👁️  VLM #{_poll_count}\n"
                    f"  Topic:    {vlm.topic} ({vlm.subtopic})\n"
                    f"  Mode:     {vlm.mode}  |  Status: {vlm.work_status}  |  Stuck: {vlm.stuck}\n"
                    f"  Screen:   {screen_details}...\n"
                    f"  Timing:   {secs_on_topic:.0f}s on topic  |  cooldown: {cooldown_left:.0f}s left\n"
                    f"  Mastery:  {bkt.get_mastery(vlm.topic):.0%} ({vlm.topic})\n"
                    f"  Stuck#:   {state.get('stuck_count', 0)}/{STUCK_OBSERVATION_COUNT}\n"
                    f"{'─' * 60}"
                )
            else:
                logger.info(
                    f"  👁️ #{_poll_count}  {vlm.mode} | {vlm.topic} | {vlm.work_status} | "
                    f"{secs_on_topic:.0f}s on topic | cd:{cooldown_left:.0f}s"
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
                logger.info(f"      ⏳ Not prompting — reason: {reason}")
                return

            # Pick the agent
            agent_name = pick_agent(vlm)
            agent_addr = state["agent_addresses"].get(agent_name)
            if not agent_addr:
                logger.warning(f"[Orchestrator] No address for {agent_name}")
                return

            # Build the request
            mastery = bkt.get_mastery(vlm.topic) if vlm.topic else 0.0
            quality = bkt.get_observation_quality(vlm.topic) if vlm.topic else {}

            # ─── DEBUG: Prompt triggered! ───
            logger.info(
                f"\n{'═' * 60}\n"
                f"  🚀 PROMPT TRIGGERED!\n"
                f"  Reason:  {reason}\n"
                f"  Agent:   {agent_name} (mode: {vlm.mode})\n"
                f"  Topic:   {vlm.topic} | Mastery: {mastery:.0%}\n"
                f"  Screen:  {screen_details}...\n"
                f"{'═' * 60}"
            )

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

            logger.info(f"  📤 Sent to {agent_name} (prompt #{state['prompt_count']})")

    except httpx.ConnectError:
        pass  # backend not running yet, that's fine
    except Exception as e:
        logger.error(f"[Orchestrator] Error: {e}")


# ─── Handle responses from agents ───────────────────────────────────
@orchestrator.on_message(model=AgentResponse)
async def handle_agent_response(ctx: Context, sender: str, msg: AgentResponse):
    """Forward agent response to the sidebar via the FastAPI backend."""
    viz_tier = (msg.metadata or {}).get("tier", "—") if msg.metadata else "—"
    logger.info(
        f"\n{'═' * 60}\n"
        f"  📥 AGENT RESPONSE RECEIVED\n"
        f"  From:     {msg.agent_type}\n"
        f"  Tool:     {msg.tool_used}  |  Type: {msg.content_type}\n"
        f"  Viz tier: {viz_tier}\n"
        f"  Content:  {msg.content[:120]}...\n"
        f"  → Forwarding to sidebar via WebSocket\n"
        f"{'═' * 60}"
    )

    # Forward to the FastAPI backend which sends it to the sidebar via WebSocket
    try:
        payload = {
            "agent_type": msg.agent_type,
            "content": msg.content,
            "content_type": msg.content_type,
            "tool_used": msg.tool_used,
            "topic": msg.topic,
            "mastery": msg.mastery,
        }
        if msg.metadata:
            payload["metadata"] = msg.metadata

        async with httpx.AsyncClient(timeout=5.0) as client:
            await client.post(
                f"{BACKEND_URL}/agent-response",
                json=payload,
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
    logger.info(f"[Orchestrator] Routes: CONCEPTUAL → conceptual | APPLIED → applied | CONSOLIDATION → extension")
    logger.info(f"[Orchestrator] Cooldown: {MIN_SECONDS_BETWEEN_PROMPTS}s | Pause: {NATURAL_PAUSE_MIN_SECONDS}s | Stuck: {STUCK_THRESHOLD_SECONDS}s | Fallback: {FALLBACK_PROMPT_SECONDS}s")
    logger.info(f"[Orchestrator] Agent address: {orchestrator.address}")
    logger.info(f"[Orchestrator] Chat Protocol: included (ASI:One discoverable)")
    logger.info(f"[Orchestrator] Payment Protocol: included (FET monetization)")
    logger.info(f"[Orchestrator] Agentverse: {'ENABLED' if AGENTVERSE_ENABLED else 'disabled'}")

    for name, addr in state["agent_addresses"].items():
        logger.info(f"[Orchestrator] {name}: {addr}")
