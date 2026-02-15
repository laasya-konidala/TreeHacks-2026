"""
Agent 1: Conceptual Understanding (Building Knowledge)

Triggered when: student is watching a video, reading notes, learning new concepts.
Goal: help them build and verify understanding of what they're consuming.

2 tools:
  - question: Socratic/comprehension question calibrated by mastery + trigger reason
  - visualization: suggest a diagram or mental model

LLM: Claude (via Anthropic API).
"""
import json
import logging
import anthropic
from uagents import Agent, Context

from agents.config import (
    ANTHROPIC_API_KEY, CLAUDE_MODEL,
    AGENTVERSE_ENABLED, AGENTVERSE_URL,
)
from agents.models import AgentRequest, AgentResponse

logger = logging.getLogger(__name__)

# ─── Agent Setup ───
CONCEPTUAL_SEED = "ambient_learning_conceptual_seed_2026"
CONCEPTUAL_PORT = 8002

conceptual_agent = Agent(
    name="conceptual_understanding",
    port=CONCEPTUAL_PORT,
    seed=CONCEPTUAL_SEED,
    endpoint=[f"http://127.0.0.1:{CONCEPTUAL_PORT}/submit"],
    agentverse=AGENTVERSE_URL if AGENTVERSE_ENABLED else None,
    mailbox=AGENTVERSE_ENABLED,
    description=(
        "Conceptual Understanding agent — helps students build knowledge "
        "by generating contextual questions and visualizations "
        "based on what they're currently watching or reading."
    ),
)

# ─── Claude Client ───
claude = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


# ─── Tool Selection Prompt ───
TOOL_SELECTION_SYSTEM = """You are a learning assistant deciding HOW to help a student who is building conceptual understanding.

You have 2 tools. Pick ONE based on the trigger reason, mastery, and what's on screen:

"question" — Ask a Socratic/comprehension question. Use when:
  - natural_pause + any mastery: check understanding of what they just saw
  - stuck: ask a simpler guiding question to unstick them
  - mode_change: bridge question connecting what they were doing to what they're doing now
  - topic_transition: ask about the connection between old and new topic
  - Low mastery: basic "what is" / definition questions
  - Medium mastery: "why" / "how does this relate" questions
  - High mastery: "what if" / extension / edge-case questions

"visualization" — Suggest a diagram or mental model. Use when:
  - The content is abstract (equations, theory, complex relationships)
  - A visual would genuinely help more than a question
  - The student has been reading/watching for a while and might benefit from a different angle

Default to "question" if unsure.

Respond with ONLY: {"tool": "question|visualization", "reasoning": "brief reason"}"""

TOOL_SELECTION_USER = """Screen details:
{screen_details}

Topic: {topic} | Mastery: {mastery}% ({mastery_quality}) | Trigger: {trigger_reason}

Recent activity:
{recent_observations}"""


# ─── Exercise Generation Prompts ───
QUESTION_SYSTEM = """You are a Socratic learning companion. Based on EXACTLY what's on the student's screen, ask ONE targeted question.

Adapt based on the trigger reason:
- natural_pause: "Before you move on..." — check they understood what they just saw
- topic_transition: "You just went from X to Y..." — connect the two
- stuck: Ask something SIMPLER to guide them — don't add pressure
- mode_change: Bridge theory ↔ practice — "Now that you're coding, how does X apply?"
- fallback: General comprehension check

Rules:
- Reference SPECIFIC things from screen_details (exact equations, code, question text, etc.)
- Ask ONE clear question, not multiple
- Don't give the answer — make them think
- Be concise and conversational, like a smart study buddy
- 2-3 sentences max"""

QUESTION_USER = """Screen details:
{screen_details}

Topic: {topic} | Mastery: {mastery}% | Trigger: {trigger_reason}

Calibrate difficulty:
- Low mastery (0-30%): "what is" / definition / basic recall
- Medium mastery (30-70%): "why does" / "how does this relate to" / reasoning
- High mastery (70-100%): "what would happen if" / "can you explain why NOT" / edge cases"""

VISUALIZATION_SYSTEM = """You are a learning companion. Suggest a specific mental visualization or diagram for what the student is looking at.

Rules:
- Describe a SPECIFIC visualization tied to what's on their screen
- Make it concrete: "Imagine..." or "Picture this..."
- If it's math: suggest a geometric interpretation or concrete example
- If it's code: suggest a flow diagram or state trace
- If it's theory: suggest an analogy from everyday life
- Keep it to 2-4 sentences
- It should help them understand, not just be decorative"""

VISUALIZATION_USER = """Screen details:
{screen_details}

Topic: {topic} | Mastery: {mastery}% | Trigger: {trigger_reason}"""

TOOL_PROMPTS = {
    "question": (QUESTION_SYSTEM, QUESTION_USER),
    "visualization": (VISUALIZATION_SYSTEM, VISUALIZATION_USER),
}


def _call_claude(system: str, user_msg: str, max_tokens: int = 300) -> str:
    """Make a Claude API call and return the text response."""
    response = claude.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user_msg}],
    )
    return response.content[0].text


def _extract_screen_details(raw_vlm_text: str, activity: str, topic: str) -> str:
    """
    Extract the specific screen_details from the raw VLM data.
    Falls back to activity + topic if screen_details is missing.
    """
    try:
        data = json.loads(raw_vlm_text) if raw_vlm_text else {}
        # screen_details comes from the VLM's detailed screen description
        details = data.get("gemini_screen_details", "")
        if details:
            return details
        # Fallback: try the screen_content field
        content = data.get("screen_content", "")
        if content:
            return content
    except Exception:
        pass
    return f"{activity} — {topic}"


# ─── Message Handler ───
@conceptual_agent.on_message(model=AgentRequest)
async def handle_request(ctx: Context, sender: str, msg: AgentRequest):
    """Handle a request from the orchestrator."""
    trigger_reason = msg.trigger_reason or "fallback"
    topic = msg.vlm_context.topic or "the current topic"
    mastery_pct = round(msg.mastery * 100)
    mastery_quality = msg.mastery_quality

    # Extract the specific screen details (not the raw JSON dump)
    screen_details = _extract_screen_details(
        msg.vlm_context.raw_vlm_text,
        msg.vlm_context.activity,
        topic,
    )

    logger.info(f"[Conceptual] Request — topic: {topic}, mastery: {mastery_pct}%, trigger: {trigger_reason}")
    logger.info(f"[Conceptual] Screen: {screen_details[:120]}")

    recent_obs = "\n".join(msg.recent_observations[-3:]) if msg.recent_observations else "No recent observations."

    # Step 1: Pick the best tool
    tool = "question"  # default
    try:
        tool_user_msg = TOOL_SELECTION_USER.format(
            screen_details=screen_details,
            topic=topic,
            mastery=mastery_pct,
            mastery_quality=mastery_quality,
            trigger_reason=trigger_reason,
            recent_observations=recent_obs,
        )

        tool_text = _call_claude(TOOL_SELECTION_SYSTEM, tool_user_msg, max_tokens=100)

        if '"visualization"' in tool_text:
            tool = "visualization"
        else:
            tool = "question"

        logger.info(f"[Conceptual] Selected tool: {tool}")

    except Exception as e:
        logger.warning(f"[Conceptual] Tool selection failed, defaulting to question: {e}")
        tool = "question"

    # Step 2: Generate the exercise using the selected tool
    try:
        system_prompt, user_template = TOOL_PROMPTS[tool]
        exercise_user_msg = user_template.format(
            screen_details=screen_details,
            topic=topic,
            mastery=mastery_pct,
            trigger_reason=trigger_reason,
        )

        content = _call_claude(system_prompt, exercise_user_msg, max_tokens=300)
        logger.info(f"[Conceptual] Generated: {content[:100]}")

    except Exception as e:
        logger.error(f"[Conceptual] Exercise generation failed: {e}")
        content = "Hmm, I wanted to ask you something about what you're reading, but I ran into an issue. Keep going!"

    # Step 3: Send response back to orchestrator
    response = AgentResponse(
        agent_type="conceptual",
        content=content,
        tool_used=tool,
        topic=topic,
        mastery=msg.mastery,
    )

    await ctx.send(sender, response)
    logger.info(f"[Conceptual] Response sent — tool: {tool}, trigger: {trigger_reason}")
