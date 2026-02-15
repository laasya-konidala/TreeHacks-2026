"""
Agent 1: Conceptual Understanding (Building Knowledge)

Triggered when: student is watching a video, reading notes, learning new concepts.
Goal: help them build and verify understanding of what they're consuming.

Uses shared tools (question, visualization, review) but frames everything
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

conceptual_agent = Agent(
    name="conceptual_understanding",
    port=CONCEPTUAL_PORT,
    seed=CONCEPTUAL_SEED,
    endpoint=[f"http://127.0.0.1:{CONCEPTUAL_PORT}/submit"],
    agentverse=AGENTVERSE_URL if AGENTVERSE_ENABLED else None,
    mailbox=AGENTVERSE_ENABLED,
    description=(
        "Conceptual Understanding agent — helps students build knowledge "
        "by generating contextual questions, visualizations, and checks "
        "based on what they're currently watching or reading."
    ),
)

# ─── Claude Client ───
claude = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

# ─── Tool Selection Prompt ───
TOOL_SELECTION_SYSTEM = """You are a learning assistant deciding HOW to help a student who is currently building conceptual understanding (watching a video, reading notes, learning something new).

Pick the BEST tool to use right now. Choose ONE:
- "question": Ask a comprehension question about what they're currently seeing
- "visualization": Suggest a way to visualize/diagram the concept to deepen understanding
- "review": Briefly summarize or connect what they're learning to something they already know

Respond with ONLY a JSON object:
{"tool": "question|visualization|review", "reasoning": "why this tool right now"}"""

TOOL_SELECTION_USER = """Current screen context:
{vlm_context}

Their mastery of "{topic}" is {mastery}% ({mastery_quality} confidence).

Recent activity:
{recent_observations}

{speech_context}"""

# ─── Exercise Generation Prompts ───
QUESTION_SYSTEM = """You are a Socratic learning companion. The student is currently watching/reading about a topic. Based on what's on their screen, ask ONE targeted comprehension question.

Rules:
- Reference EXACTLY what's on their screen (specific equations, diagrams, code, etc.)
- Ask ONE clear question, not multiple
- Don't give the answer — make them think
- Be concise and conversational, like a study buddy
- 2-3 sentences max"""

QUESTION_USER = """What's on their screen right now:
{vlm_context}

Their mastery of "{topic}" is {mastery}% — calibrate difficulty accordingly.
- Low mastery (0-30%): Ask about basic definitions or "what is" questions
- Medium mastery (30-70%): Ask "why" or "how does this relate to" questions  
- High mastery (70-100%): Ask "what would happen if" or "can you explain why NOT" questions

{speech_context}"""

VISUALIZATION_SYSTEM = """You are a learning companion. Suggest a quick mental visualization or diagram that would help the student understand what's on screen.

Rules:
- Describe a specific visualization related to what's on screen
- Make it concrete: "Imagine..." or "Picture this..."
- Keep it to 2-3 sentences
- Connect the visualization to the specific content they're looking at"""

VISUALIZATION_USER = """What's on their screen right now:
{vlm_context}

Their mastery of "{topic}" is {mastery}%.

{speech_context}"""

REVIEW_SYSTEM = """You are a learning companion. Briefly connect what the student is currently learning to something foundational or previously covered.

Rules:
- One quick connection or summary
- Reference what's specifically on their screen
- "Remember when you learned X? This builds on that because..."
- 2-3 sentences max"""

REVIEW_USER = """What's on their screen right now:
{vlm_context}

Their mastery of "{topic}" is {mastery}%.

{speech_context}"""

TOOL_PROMPTS = {
    "question": (QUESTION_SYSTEM, QUESTION_USER),
    "visualization": (VISUALIZATION_SYSTEM, VISUALIZATION_USER),
    "review": (REVIEW_SYSTEM, REVIEW_USER),
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
    tool = "question"  # default
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
        elif '"review"' in tool_text:
            tool = "review"
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
