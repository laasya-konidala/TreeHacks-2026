"""
Orchestrator Agent — Confusion detection, routing, and BKT management.
The central brain that observes, detects confusion, and deploys specialists.
"""
import asyncio
import json
import logging
import time
from typing import Optional

import httpx
from google import genai
from google.genai import types
from uagents import Agent, Context

from agents.config import (
    ORCHESTRATOR_SEED, ORCHESTRATOR_PORT, BACKEND_URL,
    INTERVENTION_COOLDOWN_SECONDS, GEMINI_MODEL, GEMINI_API_KEY,
    AGENTVERSE_ENABLED, AGENTVERSE_URL,
)
from agents.models import (
    WorkContext, ConfusionAssessment,
    DeepDiveRequest, VisualizerRequest, AssessorRequest,
    AgentMessage, SessionReport, UserReply,
)
from agents.confusion_detector import ConfusionDetector
from agents.learner_model import ConfidenceWeightedBKT
from agents.observation_pipeline import ObservationPipeline

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
        "Ambient learning orchestrator — observes user work and deploys "
        "specialist tutoring agents when confusion is detected. Uses "
        "Bayesian Knowledge Tracing with confidence weighting to track "
        "learner progress across concepts."
    ),
    publish_agent_details=AGENTVERSE_ENABLED,
)

# ─── Include Protocols (Chat for ASI:One, Payment for monetization) ───
from agents.chat_protocol import chat_dialogue
from agents.payment_protocol import payment_protocol

orchestrator.include(chat_dialogue, publish_manifest=True)
orchestrator.include(payment_protocol, publish_manifest=True)

# ─── State ───
bkt = ConfidenceWeightedBKT()
pipeline = ObservationPipeline(bkt)
detector = ConfusionDetector()
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

state = {
    "active_dialogue": None,         # current session_id if any
    "last_intervention": 0.0,        # timestamp for cooldown
    "last_screen_analysis": 0.0,     # rate limit screen analysis
    "pending_assessor": None,        # concept to send to assessor after dialogue
    "agent_addresses": {
        "deep_diver": None,
        "assessor": None,
        "visualizer": None,
    },
}


def _resolve_agent_addresses():
    """Resolve agent addresses from seeds (deterministic)."""
    from uagents import Agent as _Agent

    # Create temporary agents to get addresses from seeds
    from agents.config import DEEP_DIVER_SEED, ASSESSOR_SEED, VISUALIZER_SEED

    _dd = _Agent(name="_dd", seed=DEEP_DIVER_SEED)
    _as = _Agent(name="_as", seed=ASSESSOR_SEED)
    _vi = _Agent(name="_vi", seed=VISUALIZER_SEED)

    state["agent_addresses"]["deep_diver"] = _dd.address
    state["agent_addresses"]["assessor"] = _as.address
    state["agent_addresses"]["visualizer"] = _vi.address

    logger.info(f"Resolved addresses: {state['agent_addresses']}")


@orchestrator.on_event("startup")
async def startup(ctx: Context):
    """Initialize agent addresses on startup."""
    _resolve_agent_addresses()
    logger.info(f"Orchestrator started: {orchestrator.address}")


@orchestrator.on_interval(period=2.0)
async def poll_context(ctx: Context):
    """Poll for new WorkContext from the FastAPI server."""
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{BACKEND_URL}/context/latest", timeout=2.0)
            data = resp.json()
    except Exception as e:
        return  # Server not ready yet or no data

    if not data:
        return

    # Parse WorkContext (merges Gemini VLM + Chrome extension behavioral data)
    try:
        work_ctx = WorkContext(**{
            "screen_content": data.get("screen_content", ""),
            "screen_content_type": data.get("screen_content_type", "text"),
            "detected_topic": data.get("detected_topic", ""),
            "detected_subtopic": data.get("detected_subtopic", ""),
            "typing_speed_ratio": float(data.get("typing_speed_ratio", 1.0)),
            "deletion_rate": float(data.get("deletion_rate", 0.0)),
            "pause_duration": float(data.get("pause_duration", 0.0)),
            "scroll_back_count": int(data.get("scroll_back_count", 0)),
            "audio_transcript": data.get("audio_transcript"),
            "verbal_confusion_cues": data.get("verbal_confusion_cues", []),
            "user_touched_agent": bool(data.get("user_touched_agent", False)),
            "user_message": data.get("user_message"),
            "screenshot_b64": data.get("screenshot_b64"),
            "user_id": data.get("user_id", "default"),
            "session_id": data.get("session_id", ""),
            "timestamp": data.get("timestamp", ""),
            # Gemini VLM fields
            "gemini_stuck": bool(data.get("gemini_stuck", False)),
            "gemini_work_status": data.get("gemini_work_status", "unclear"),
            "gemini_confused_about": data.get("gemini_confused_about", []),
            "gemini_understands": data.get("gemini_understands", []),
            "gemini_error": data.get("gemini_error"),
            "gemini_mode": data.get("gemini_mode", ""),
            "gemini_notes": data.get("gemini_notes", ""),
        })
    except Exception as e:
        logger.warning(f"Failed to parse WorkContext: {e}")
        return

    # If active dialogue, check for user reply instead of re-triggering
    if state["active_dialogue"]:
        await _check_for_reply(ctx)
        return

    # Run confusion detection (now uses Gemini VLM signals + behavioral)
    assessment = detector.score(work_ctx)

    # Use Gemini-provided topic, fall back to keyword inference
    topic = work_ctx.detected_topic or _infer_topic(work_ctx.screen_content)

    # Process behavioral observations into BKT
    if topic:
        pipeline.process_behavioral(
            topic,
            work_ctx.typing_speed_ratio,
            work_ctx.deletion_rate,
            work_ctx.pause_duration,
        )

    # Process Gemini VLM observations into BKT (replaces Claude Vision)
    if topic and (work_ctx.gemini_understands or work_ctx.gemini_confused_about):
        _process_gemini_observations(topic, work_ctx)

    # Check if we should intervene
    if not assessment.should_intervene:
        return

    # Payment tier check — enforce daily intervention limits
    from agents.payment_protocol import check_can_intervene, record_intervention
    user_id = work_ctx.user_id or "default"
    if not check_can_intervene(user_id):
        logger.info(f"User {user_id} hit intervention limit for their tier")
        return

    # Cooldown check
    now = time.time()
    if now - state["last_intervention"] < INTERVENTION_COOLDOWN_SECONDS:
        return

    state["last_intervention"] = now
    record_intervention(user_id)
    logger.info(f"Intervention triggered: {assessment.reasoning}")

    # Route to specialist
    await _route_to_agent(ctx, work_ctx, assessment)


async def _check_for_reply(ctx: Context):
    """Check for pending user replies to forward to active dialogue agent."""
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{BACKEND_URL}/reply/poll", timeout=2.0)
            data = resp.json()
    except Exception:
        return

    if not data or not data.get("message"):
        return

    # Forward reply to deep diver
    dd_addr = state["agent_addresses"]["deep_diver"]
    if dd_addr:
        await ctx.send(
            dd_addr,
            UserReply(
                message=data["message"],
                session_id=state["active_dialogue"],
                user_id=data.get("user_id", "default"),
            ),
        )


async def _route_to_agent(ctx: Context, work_ctx: WorkContext, assessment: ConfusionAssessment):
    """Route to the appropriate specialist agent."""
    confusion_type = assessment.confusion_type
    topic = work_ctx.detected_topic or _infer_topic(work_ctx.screen_content)
    session_id = f"session_{int(time.time())}_{work_ctx.user_id}"

    if confusion_type == "CONCEPTUAL_WHY" or confusion_type == "EXPLICIT_REQUEST":
        # Deep Diver — multi-turn dialogue
        addr = state["agent_addresses"]["deep_diver"]
        if addr:
            state["active_dialogue"] = session_id
            await ctx.send(
                addr,
                DeepDiveRequest(
                    concept=topic or "the concept",
                    confusion_hypothesis=_build_hypothesis(work_ctx, assessment),
                    screen_content=work_ctx.screen_content[:1000],
                    screen_content_type=work_ctx.screen_content_type,
                    user_message=work_ctx.user_message,
                    user_id=work_ctx.user_id,
                    session_id=session_id,
                ),
            )

    elif confusion_type == "VISUAL_SPATIAL":
        # Visualizer
        addr = state["agent_addresses"]["visualizer"]
        if addr:
            await ctx.send(
                addr,
                VisualizerRequest(
                    concept=topic or "the concept",
                    subconcept=work_ctx.detected_subtopic or "",
                    screen_content=work_ctx.screen_content[:1000],
                    confusion_hypothesis=_build_hypothesis(work_ctx, assessment),
                    user_id=work_ctx.user_id,
                    session_id=session_id,
                ),
            )

    elif confusion_type == "NONE_EXTENDING":
        # Assessor — contrastive challenge
        addr = state["agent_addresses"]["assessor"]
        if addr:
            mastery = bkt.get_mastery(topic) if topic else 0.5
            await ctx.send(
                addr,
                AssessorRequest(
                    concept=topic or "the concept",
                    user_solution=work_ctx.screen_content[:1000],
                    screen_content_type=work_ctx.screen_content_type,
                    mastery_level=mastery,
                    user_id=work_ctx.user_id,
                    session_id=session_id,
                ),
            )

    elif confusion_type == "PROCEDURAL_HOW":
        # Inline hint via Gemini API (no specialist agent)
        hint = _generate_procedural_hint(topic, work_ctx.screen_content)
        await _broadcast_response(AgentMessage(
            content=hint,
            content_type="hint",
            agent_type="orchestrator",
            session_id=session_id,
            metadata={"concept": topic},
        ))


@orchestrator.on_message(model=AgentMessage)
async def handle_agent_response(ctx: Context, sender: str, msg: AgentMessage):
    """Forward agent responses to UI via FastAPI WebSocket broadcast."""
    await _broadcast_response(msg)


@orchestrator.on_message(model=SessionReport)
async def handle_session_report(ctx: Context, sender: str, msg: SessionReport):
    """Handle dialogue session completion."""
    logger.info(
        f"Session complete: {msg.session_id}, concept={msg.concept}, "
        f"comprehension={msg.final_comprehension}, reason={msg.close_reason}"
    )

    # Clear active dialogue
    state["active_dialogue"] = None

    # Update BKT with dialogue observations
    pipeline.process_observations(msg.observations)

    # If comprehension was high, schedule assessor
    if msg.final_comprehension > 0.6:
        state["pending_assessor"] = msg.concept
        # Will be triggered on next poll cycle if conditions met


async def _broadcast_response(msg: AgentMessage):
    """Send agent response to FastAPI for WebSocket broadcast."""
    try:
        async with httpx.AsyncClient() as client:
            await client.post(
                f"{BACKEND_URL}/agent-response",
                json={
                    "content": msg.content,
                    "content_type": msg.content_type,
                    "agent_type": msg.agent_type,
                    "dialogue_state": msg.dialogue_state,
                    "session_id": msg.session_id,
                    "turn_number": msg.turn_number,
                    "metadata": msg.metadata,
                },
                timeout=5.0,
            )
    except Exception as e:
        logger.error(f"Failed to broadcast response: {e}")


def _infer_topic(screen_content: str) -> str:
    """Infer topic from screen content (simple keyword matching)."""
    if not screen_content:
        return ""

    content_lower = screen_content.lower()
    topic_keywords = {
        "linear_regression": ["linear regression", "least squares", "y = mx + b", "slope", "intercept"],
        "gradient_descent": ["gradient descent", "learning rate", "gradient", "optimization", "convergence"],
        "neural_network": ["neural network", "neural net", "hidden layer", "activation function", "backpropagation"],
        "regularization": ["regularization", "l1", "l2", "lasso", "ridge", "overfitting"],
        "probability": ["probability", "bayes", "distribution", "likelihood", "prior", "posterior"],
        "calculus": ["derivative", "integral", "differentiation", "limit", "chain rule"],
        "linear_algebra": ["matrix", "vector", "eigenvalue", "determinant", "linear transformation"],
        "statistics": ["mean", "variance", "standard deviation", "hypothesis test", "p-value"],
        "classification": ["classification", "logistic regression", "decision boundary", "softmax"],
        "clustering": ["clustering", "k-means", "centroid", "unsupervised"],
    }

    for topic, keywords in topic_keywords.items():
        for kw in keywords:
            if kw in content_lower:
                return topic

    return "general"


def _process_gemini_observations(topic: str, work_ctx: WorkContext):
    """
    Feed Gemini VLM analysis into BKT model.
    Gemini tells us what the user understands and what they're confused about.
    """
    # Concepts Gemini says user understands → correct observations
    for concept in work_ctx.gemini_understands:
        concept_id = concept.lower().replace(" ", "_")
        bkt.update(concept_id, correct=True, confidence=0.7)

    # Concepts Gemini says user is confused about → incorrect observations
    for concept in work_ctx.gemini_confused_about:
        concept_id = concept.lower().replace(" ", "_")
        bkt.update(concept_id, correct=False, confidence=0.7)

    # If Gemini detected incorrect work on the main topic
    if work_ctx.gemini_work_status == "incorrect":
        bkt.update(topic, correct=False, confidence=0.8)
    elif work_ctx.gemini_work_status == "correct":
        bkt.update(topic, correct=True, confidence=0.8)


def _build_hypothesis(work_ctx: WorkContext, assessment: ConfusionAssessment) -> str:
    """Build a confusion hypothesis from context, assessment, and Gemini analysis."""
    parts = []

    # Gemini analysis (most informative)
    if work_ctx.gemini_confused_about:
        parts.append(f"Gemini detected confusion about: {', '.join(work_ctx.gemini_confused_about)}")
    if work_ctx.gemini_error:
        parts.append(f"Error in work: {work_ctx.gemini_error}")
    if work_ctx.gemini_stuck:
        parts.append("Gemini detected user is stuck")
    if work_ctx.gemini_work_status == "incorrect":
        parts.append("Work appears incorrect")
    if work_ctx.gemini_notes:
        parts.append(f"Gemini notes: {work_ctx.gemini_notes}")

    if work_ctx.user_message:
        parts.append(f"User said: '{work_ctx.user_message}'")
    if assessment.confusion_type:
        parts.append(f"Confusion type: {assessment.confusion_type}")

    top_signals = sorted(
        assessment.signals.items(), key=lambda x: x[1], reverse=True
    )[:2]
    signal_desc = []
    for name, val in top_signals:
        if val > 0.3:
            if name == "pause":
                signal_desc.append("long pauses (stuck)")
            elif name == "deletion":
                signal_desc.append("lots of backspacing (trial and error)")
            elif name == "reread":
                signal_desc.append("re-reading content (not understanding)")
            elif name == "typing":
                signal_desc.append("slow typing (hesitating)")
            elif name == "gemini":
                signal_desc.append("Gemini VLM detected confusion")
    if signal_desc:
        parts.append(f"Behavioral signals: {', '.join(signal_desc)}")

    return "; ".join(parts) if parts else "general confusion detected"


def _generate_procedural_hint(topic: str, screen_content: str) -> str:
    """Generate a quick procedural hint via Gemini API."""
    try:
        response = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=f"Student is working on {topic}. Their current work:\n{screen_content[:500]}\n\nGive a quick procedural hint.",
            config=types.GenerateContentConfig(
                system_instruction="You're a helpful tutor giving a quick hint. Be concise — 1-2 sentences max. Don't solve it, just point them in the right direction.",
                max_output_tokens=150,
            ),
        )
        return response.text or ""
    except Exception as e:
        logger.error(f"Hint generation failed: {e}")
        return f"💡 Try breaking the {topic} problem into smaller steps. What's the first thing you need to figure out?"
