"""
Agent: Applied (Problem Solving & Agentic Reasoning)

Triggered when: student is actively solving a problem, writing code,
working through exercises, or applying concepts hands-on.

Goal: guide the student through problem-solving by
scaffolding their reasoning process, working iteratively and incrementally to get to a solution, not giving answers

Uses shared tools (voice_call, visualization) but frames everything
through a "how are you approaching this problem?" lens, and help them think through the problem incrementally and iteratively. Being clear about the goal being able to solve a problem. 

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
APPLIED_SEED = "ambient_learning_applied_seed_2026"
APPLIED_PORT = 8003

applied_agent = Agent(
    name="applied_problem_solving",
    port=APPLIED_PORT,
    seed=APPLIED_SEED,
    endpoint=[f"http://127.0.0.1:{APPLIED_PORT}/submit"],
    agentverse=AGENTVERSE_URL if AGENTVERSE_ENABLED else None,
    mailbox=AGENTVERSE_ENABLED,
    description=(
        "Applied problem-solving agent — helps students work through problems "
        "by scaffolding their reasoning process, identifying stuck points, "
        "and guiding them toward solutions without giving answers directly."
    ),
)

# ─── Claude Client ───
claude = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

# ─── Tool Selection Prompt ───
#
# INSERT updated reasoning for how this agent decides between tools.
# The framing should be problem-solving oriented — e.g., "the student
# is actively working on something, how do we best support them?"
#
TOOL_SELECTION_SYSTEM = """<INSERT TOOL SELECTION SYSTEM PROMPT — e.g., You are a learning
assistant deciding HOW to help a student who is actively solving a problem.
Consider whether they need to talk through their approach or see the
problem from a different angle.>

Pick the BEST tool to use right now. Choose ONE:
- "voice_call": Initiate a spoken dialogue with the student to talk through their problem-solving approach
- "visualization": Suggest a way to visualize/diagram the problem to unblock their thinking

Respond with ONLY a JSON object:
{"tool": "voice_call|visualization", "reasoning": "why this tool right now"}"""

TOOL_SELECTION_USER = """Current screen context:
{vlm_context}

Their mastery of "{topic}" is {mastery}% ({mastery_quality} confidence).

Recent activity:
{recent_observations}

{speech_context}"""

# ─── Exercise Generation Prompts ───
#
# INSERT updated reasoning for how voice_call works in a problem-solving
# context. The current prompts below are placeholders — replace the
# VOICE_CALL_SYSTEM content with applied/problem-solving specific logic.
#
VOICE_CALL_SYSTEM = """< You are a friendly spoken-word
tutor helping a student who is in the middle of solving a problem. Based on
what's on their screen, generate an opening line that helps them reason
through their current step and future steps on this problemwithout giving the answer away.>

Rules:
- Reference EXACTLY what's on their screen (specific equations, code, work-in-progress, etc.)
- Do not solve the problem for them
- Sound natural and spoken — this will be read aloud, not displayed as text
- Open with ONE clear thought or question about their current approach
- Keep it to 2-3 sentences max
- Do NOT give incorrect information, that is worse than giving no information at all"""

VOICE_CALL_USER = """What's on their screen right now:
{vlm_context}

Their mastery of "{topic}" is {mastery}% — calibrate difficulty accordingly.
- Low mastery (0-30%): <INSERT — e.g., Help them identify the first step>
- Medium mastery (30-70%): <INSERT — e.g., Ask about their strategy or what they've tried>
- High mastery (70-100%): <INSERT — e.g., Challenge them on efficiency or alternative approaches>

{speech_context}"""

VISUALIZATION_SYSTEM = """You are a learning companion. Suggest a visualization or diagram that would help the student understand what's on screen.

Rules:
- Describe a specific visualization related to what's on screen and specifically what is related to the problem at hand instead of an arbitrary simplification or extension of some other concept
- Let elements of the visualization be adjustable if elements of the problem at hand are adjustable and relevant to the problem at hand
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
@applied_agent.on_message(model=AgentRequest)
async def handle_request(ctx: Context, sender: str, msg: AgentRequest):
    """Handle a request from the orchestrator."""
    logger.info(f"[Applied] Received request — topic: {msg.vlm_context.topic}, mastery: {msg.mastery:.0%}")

    # ── Unpack the incoming context (same for every agent) ──
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

        logger.info(f"[Applied] Selected tool: {tool}")

    except Exception as e:
        logger.warning(f"[Applied] Tool selection failed, defaulting to voice_call: {e}")
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
        logger.info(f"[Applied] Generated: {content[:100]}")

    except Exception as e:
        logger.error(f"[Applied] Exercise generation failed: {e}")
        content = "I wanted to help you think through this problem, but hit a snag. Keep working — you've got this!"

    # ── Step 3: Send response back to orchestrator (same for every agent) ──
    response = AgentResponse(
        agent_type="applied",
        content=content,
        tool_used=tool,
        topic=topic,
        mastery=msg.mastery,
    )

    await ctx.send(sender, response)
    logger.info(f"[Applied] Response sent — tool: {tool}")
