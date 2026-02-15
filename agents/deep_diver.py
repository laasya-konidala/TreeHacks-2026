"""
Deep Diver Agent — Multi-turn conceptual explanation via dialogue state machine.
The most complex and impressive agent for demo purposes.
"""
import json
import logging
import time
from typing import Optional

from google import genai
from google.genai import types
from uagents import Agent, Context

from agents.config import (
    DEEP_DIVER_SEED, DEEP_DIVER_PORT, GEMINI_MODEL, GEMINI_API_KEY,
    AGENTVERSE_ENABLED, AGENTVERSE_URL,
)
from agents.models import (
    DeepDiveRequest, DeepDiveResponse, UserReply,
    AgentMessage, SessionReport,
)
from agents.dialogue_session import DialogueSession

logger = logging.getLogger(__name__)

# ─── Agent Setup ───
deep_diver = Agent(
    name="concept_deep_diver",
    port=DEEP_DIVER_PORT,
    seed=DEEP_DIVER_SEED,
    endpoint=[f"http://127.0.0.1:{DEEP_DIVER_PORT}/submit"],
    agentverse=AGENTVERSE_URL if AGENTVERSE_ENABLED else None,
    mailbox=AGENTVERSE_ENABLED,
    description=(
        "Conceptual explanation agent — multi-turn Socratic dialogue for "
        "deep understanding of technical concepts. Uses a state machine "
        "(initiating → exploring → explaining → checking → closing) to "
        "guide the conversation and track comprehension."
    ),
    publish_agent_details=AGENTVERSE_ENABLED,
)

# ─── State ───
active_sessions: dict[str, DialogueSession] = {}
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

# ─── Prompt Templates ───
PROMPTS = {
    "initiating": (
        "You noticed the student is working on {concept} and seems stuck on "
        "{confusion}. Open casually — mention what you noticed, ask a brief open "
        "question. ONE short paragraph. Sound like a smart friend, not a professor."
    ),
    "exploring": (
        "The student said: '{reply}'. You're still diagnosing their specific "
        "confusion about {concept}. Ask ONE targeted follow-up question. Don't "
        "explain yet — understand first. 1-2 sentences."
    ),
    "explaining": (
        "Now explain. The student understands: {confirmed}. They're confused about: "
        "{hypothesis}. DO NOT use these approaches (already tried): {tried}. "
        "Bridge from what they know. Use ONE concrete analogy or example. "
        "Max 2 paragraphs. End with something that invites response."
    ),
    "checking": (
        "The student seems to understand {concept}. Ask them to restate the key "
        "idea in their own words, OR give a quick check question. Make it "
        "conversational, not a quiz. 1-2 sentences."
    ),
    "closing": (
        "They've got it — {concept}. Summarize the key insight in ONE sentence. "
        "Say you'll let them get back to work. Warm and brief."
    ),
}

ANALYSIS_PROMPT = (
    "Analyze this student response during a tutoring session.\n"
    "Context: confused about {concept}.\n"
    "Dialogue so far:\n{history}\n\n"
    "Latest student response: '{reply}'\n\n"
    "Return ONLY valid JSON (no markdown fences):\n"
    '{{"comprehension": 0.0 to 1.0, "restated_in_own_words": true/false, '
    '"remaining_confusion": null or "description", '
    '"misconception_detected": null or "description", '
    '"engagement_level": "high" or "medium" or "low"}}'
)


def _call_gemini(system_msg: str, user_msg: str, max_tokens: int = 300) -> str:
    """Make a Gemini API call."""
    try:
        response = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=user_msg,
            config=types.GenerateContentConfig(
                system_instruction=system_msg,
                max_output_tokens=max_tokens,
            ),
        )
        return response.text or ""
    except Exception as e:
        logger.error(f"Gemini API error: {e}")
        return "I'm having trouble connecting right now. Let me try again in a moment."


def _analyze_response(concept: str, history: str, reply: str) -> dict:
    """Analyze a user's response using Gemini."""
    prompt = ANALYSIS_PROMPT.format(
        concept=concept,
        history=history,
        reply=reply,
    )
    try:
        response = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=200,
            ),
        )
        text = (response.text or "").replace("```json", "").replace("```", "").strip()
        return json.loads(text)
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return {
            "comprehension": 0.5,
            "restated_in_own_words": False,
            "remaining_confusion": None,
            "misconception_detected": None,
            "engagement_level": "medium",
        }


def _generate_turn(session: DialogueSession, user_reply: str = "") -> str:
    """Generate the next agent turn based on dialogue state."""
    state = session.state
    cm = session.confusion_model

    template = PROMPTS.get(state, PROMPTS["exploring"])

    prompt = template.format(
        concept=session.concept,
        confusion=cm.get("initial_hypothesis", "something"),
        reply=user_reply,
        confirmed=", ".join(cm.get("confirmed_understanding", [])) or "nothing confirmed yet",
        hypothesis=cm.get("refined_hypothesis") or cm.get("initial_hypothesis", "the concept"),
        tried=", ".join(cm.get("approaches_tried", [])) or "none yet",
    )

    system_msg = (
        "You are a friendly, knowledgeable tutor. You're helping a student who is "
        "working and got stuck. Be conversational, warm, and concise. Never be "
        "condescending. Use simple language and concrete examples."
    )

    return _call_gemini(system_msg, prompt)


@deep_diver.on_message(model=DeepDiveRequest)
async def handle_initial(ctx: Context, sender: str, msg: DeepDiveRequest):
    """Handle initial deep dive request — create session and generate opening."""
    session_id = msg.session_id or f"dd_{int(time.time())}_{msg.user_id}"

    # Create dialogue session
    session = DialogueSession(
        session_id=session_id,
        user_id=msg.user_id,
        trigger_context={
            "concept": msg.concept,
            "confusion_hypothesis": msg.confusion_hypothesis,
            "screen_content": msg.screen_content,
            "screen_content_type": msg.screen_content_type,
        },
    )

    # Generate opening message
    opening = _generate_turn(session)
    session.add_agent_turn(opening)
    session.advance_state()  # initiating → exploring

    # Store session
    active_sessions[session_id] = session

    logger.info(f"Deep dive started: session={session_id}, concept={msg.concept}")

    # Send response back to orchestrator
    await ctx.send(
        sender,
        AgentMessage(
            content=opening,
            content_type="text",
            agent_type="deep_diver",
            dialogue_state=session.state,
            session_id=session_id,
            turn_number=1,
            metadata={"concept": msg.concept},
        ),
    )


@deep_diver.on_message(model=UserReply)
async def handle_reply(ctx: Context, sender: str, msg: UserReply):
    """Handle user reply in an active dialogue session."""
    session = active_sessions.get(msg.session_id)
    if not session:
        logger.warning(f"No active session: {msg.session_id}")
        await ctx.send(
            sender,
            AgentMessage(
                content="I seem to have lost track of our conversation. Could you ask again?",
                content_type="text",
                agent_type="deep_diver",
                session_id=msg.session_id,
                turn_number=0,
            ),
        )
        return

    # Analyze user response
    history = session.get_dialogue_for_prompt()
    analysis = _analyze_response(session.concept, history, msg.message)

    # Record user turn with analysis
    session.add_user_turn(msg.message, analysis)

    # Update confusion model
    if analysis.get("remaining_confusion"):
        session.confusion_model["refined_hypothesis"] = analysis["remaining_confusion"]

    # Advance state machine
    session.advance_state()

    # Generate next turn
    response_text = _generate_turn(session, msg.message)
    session.add_agent_turn(response_text)

    # Track what approaches we've tried
    session.confusion_model["approaches_tried"].append(
        f"Turn {session.turn_count}: {session.state}"
    )

    # Check if closing
    is_closing = session.state == "closing" or session.should_close()

    if is_closing:
        # Send session report
        close_reason = session.get_close_reason()
        await ctx.send(
            sender,
            SessionReport(
                session_id=session.session_id,
                user_id=session.user_id,
                concept=session.concept,
                turns_count=session.turn_count,
                final_comprehension=session.get_final_comprehension(),
                observations=session.get_observations(),
                duration_seconds=round(session.elapsed, 1),
                close_reason=close_reason,
            ),
        )
        # Clean up
        del active_sessions[session.session_id]
        logger.info(f"Deep dive closed: session={session.session_id}, reason={close_reason}")

    # Send response
    await ctx.send(
        sender,
        AgentMessage(
            content=response_text,
            content_type="text",
            agent_type="deep_diver",
            dialogue_state=session.state,
            session_id=session.session_id,
            turn_number=session.turn_count,
            metadata={
                "concept": session.concept,
                "comprehension": analysis.get("comprehension", 0.5),
                "should_close": is_closing,
            },
        ),
    )
