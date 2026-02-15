"""
Agent 1: Conceptual Understanding (Building Knowledge)

Triggered when: student is watching a video, reading notes, learning new concepts.
Goal: help them build and verify understanding of what they're consuming.

Uses shared tools (voice_call, visualization) but frames everything
through a "do you understand what you're seeing?" lens.

LLM: Claude (via Anthropic API) for high-quality exercise generation.
"""
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

_agent_kwargs = dict(
    name="conceptual_understanding",
    port=CONCEPTUAL_PORT,
    seed=CONCEPTUAL_SEED,
    description=(
        "Conceptual Understanding agent — helps students build knowledge "
        "by generating contextual questions, visualizations, and checks "
        "based on what they're currently watching or reading."
    ),
)
if AGENTVERSE_ENABLED:
    _agent_kwargs["mailbox"] = True
else:
    _agent_kwargs["endpoint"] = [f"http://127.0.0.1:{CONCEPTUAL_PORT}/submit"]

conceptual_agent = Agent(**_agent_kwargs)

# ─── Claude Client ───
claude = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

# ─── Tool Selection Prompt ───
TOOL_SELECTION_SYSTEM = """You are a learning assistant deciding HOW to help a student who is currently building conceptual understanding (watching a video, reading notes, learning something new).

Pick the BEST tool to use right now. Choose ONE:
- "voice_call": Initiate a spoken dialogue with the student to talk through the concept conversationally
- "visualization": Suggest a way to visualize/diagram the concept to deepen understanding

Respond with ONLY a JSON object:
{"tool": "voice_call|visualization", "reasoning": "why this tool right now"}"""

TOOL_SELECTION_USER = """Current screen context:
{vlm_context}

Their mastery of "{topic}" is {mastery}% ({mastery_quality} confidence).

Recent activity:
{recent_observations}

{speech_context}"""

# ─── Exercise Generation Prompts ───
VOICE_CALL_SYSTEM = """You are a friendly spoken-word tutor about to start a live voice conversation with the student to explore the deep conceptual understanding of the topic at hand .
 Based on what's on their screen, generate an opening line that kicks off a short dialogue about the concept.

Rules:
- Reference EXACTLY what's on their screen (specific equations, diagrams, code, etc.)
- Do not use any novel or new concepts beyond the level of understanding of the current concept in front of the student
- Sound natural and spoken — this will be read aloud, not displayed as text
- Open with ONE clear thought or question to get them talking
- Don't give the answer — make them think, but do not arbitrarily challenge them
- Keep it to 2-3 sentences max
- Do NOT give incorrect information, that is worse than giving no information at all"""

VOICE_CALL_USER = """What's on their screen right now:
{vlm_context}

Their mastery of "{topic}" is {mastery}% — calibrate difficulty accordingly.
- Low mastery (0-30%): Start with a gentle check-in about the basics
- Medium mastery (30-70%): Ask them to talk through their reasoning
- High mastery (70-100%): Challenge them to explain an edge case or tradeoff out loud

{speech_context}"""

VISUALIZATION_SYSTEM = """You are a learning companion. Suggest a visualization or diagram that would help the student understand what's on screen.

Rules:
- Describe a specific visualization related to what's on screen
- Keep it easily extendable to their current concept but able to adjust the specific variables of whats going on
- Make it concrete: "Imagine..." or "Picture this..."
- Keep it to 2-3 sentences
- Do NOT give incorrect information, that is worse than giving no informtion at all
- Connect the visualization to the specific content they're looking at"""

VISUALIZATION_USER = """What's on their screen right now:
{vlm_context}

Their mastery of "{topic}" is {mastery}%.

{speech_context}"""

TOOL_PROMPTS = {
    "voice_call": (VOICE_CALL_SYSTEM, VOICE_CALL_USER),
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


# ─── Message Handler ───
@conceptual_agent.on_message(model=AgentRequest)
async def handle_request(ctx: Context, sender: str, msg: AgentRequest):
    """Handle a request from the orchestrator."""
    logger.info(f"[Conceptual] Received request — topic: {msg.vlm_context.topic}, mastery: {msg.mastery:.0%}")

    vlm_text = msg.vlm_context.raw_vlm_text or f"{msg.vlm_context.activity} — {msg.vlm_context.topic}"
    topic = msg.vlm_context.topic or "the current topic"
    mastery_pct = round(msg.mastery * 100)
    mastery_quality = msg.mastery_quality

    speech_context = ""
    if msg.vlm_context.speech_transcript:
        speech_context = f'The student just said: "{msg.vlm_context.speech_transcript}"'

    recent_obs = "\n".join(msg.recent_observations[-3:]) if msg.recent_observations else "No recent observations."

    # Step 1: Pick the best tool
    tool = "voice_call"  # default
    try:
        tool_user_msg = TOOL_SELECTION_USER.format(
            vlm_context=vlm_text,
            topic=topic,
            mastery=mastery_pct,
            mastery_quality=mastery_quality,
            recent_observations=recent_obs,
            speech_context=speech_context,
        )

        tool_text = _call_claude(TOOL_SELECTION_SYSTEM, tool_user_msg, max_tokens=100)

        if '"visualization"' in tool_text:
            tool = "visualization"
        else:
            tool = "voice_call"

        logger.info(f"[Conceptual] Selected tool: {tool}")

    except Exception as e:
        logger.warning(f"[Conceptual] Tool selection failed, defaulting to voice_call: {e}")
        tool = "voice_call"

    # Step 2: Generate the exercise using the selected tool
    try:
        system_prompt, user_template = TOOL_PROMPTS[tool]
        exercise_user_msg = user_template.format(
            vlm_context=vlm_text,
            topic=topic,
            mastery=mastery_pct,
            speech_context=speech_context,
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
    logger.info(f"[Conceptual] Response sent — tool: {tool}")
