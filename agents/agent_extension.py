"""
Agent: Extension (Stretch & Connect)

Triggered when: student has demonstrated solid understanding and is ready
to go beyond the current material.

Goal: push the student to transfer, generalize, or connect the concept
to adjacent ideas they haven't seen yet.

Uses shared tools (voice_call, visualization) but frames everything
through a "can you go further / connect this to something new?" lens.

LLM: Claude (via Anthropic API) for high-quality exercise generation.
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
EXTENSION_SEED = "ambient_learning_extension_seed_2026"
EXTENSION_PORT = 8004

_agent_kwargs = dict(
    name="extension_stretch",
    port=EXTENSION_PORT,
    seed=EXTENSION_SEED,
    description=(
        "Extension agent — pushes students beyond current material by generating "
        "transfer questions, cross-topic connections, and stretch challenges "
        "based on demonstrated mastery. Takes a topic to a new abstraction of knowledge."
    ),
)
if AGENTVERSE_ENABLED:
    _agent_kwargs["mailbox"] = True
else:
    _agent_kwargs["endpoint"] = [f"http://127.0.0.1:{EXTENSION_PORT}/submit"]

extension_agent = Agent(**_agent_kwargs)

# ─── Claude Client ───
claude = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

# ─── Tool Selection Prompt ───
TOOL_SELECTION_SYSTEM = """You are a learning assistant deciding HOW to help a student who has demonstrated solid understanding and is ready to stretch beyond the current material.

Pick the BEST tool to use right now. Choose ONE:

"visualization" — Generate a diagram or interactive visual. STRONGLY PREFERRED. Use when:
  - You can show how the current concept extends or generalizes visually
  - There are related concepts that connect through a diagram or graph
  - You want to show what happens when you change a variable or extend a formula
  - The extension involves spatial, structural, or mathematical relationships
  - A side-by-side comparison or "what if" diagram would deepen understanding
  - You can visualize the bridge between the current topic and the extension

"voice_call" — Start a spoken dialogue. Use ONLY when:
  - The extension is purely philosophical or definitional
  - You want a Socratic dialogue about big-picture connections with no visual component
  - The screen content is too vague to create a meaningful visualization

Default to "visualization" if unsure — visuals make abstract extensions concrete and memorable.

Respond with ONLY a JSON object:
{"tool": "voice_call|visualization", "reasoning": "why this tool right now"}"""

TOOL_SELECTION_USER = """Current screen context:
{vlm_context}

Their mastery of "{topic}" is {mastery}% ({mastery_quality} confidence).

Recent activity:
{recent_observations}

{speech_context}"""

# ─── Exercise Generation Prompts ───
VOICE_CALL_SYSTEM = """You are a friendly spoken-word tutor about to start a live voice conversation with the student. 
The student has strong understanding of the topic at hand and is ready to go beyond the current material, at a level of abstraction one away or two away from the current concept. YOu want to make jumps in reasoning logical and fluid. 
 Based on what's on their screen, generate an opening line that kicks off a short dialogue about the concept.

Rules:
- Reference concepts in the context of what's on their screen (specific equations, diagrams, code, etc.)
- When making extensions, make sure to make logicial jumps in reasoning that follow from the original content
- Sound natural and spoken — this will be read aloud, not displayed as text
- Open with ONE clear thought or question to get them talking, 
- Don't give the answer, and do not make the extension trivial, but do not make it too difficult as well— make them think, but do not arbitrarily challenge them
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
- make the extension soemthign that leads to adjustments of a particular feature, that points to a new classification of the concept or a new abstraction of the concept 
- Make sure it is an intuitive and logitical jump in reasoning that can be visually displayed
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
@extension_agent.on_message(model=AgentRequest)
async def handle_request(ctx: Context, sender: str, msg: AgentRequest):
    """Handle a request from the orchestrator."""
    logger.info(f"[Extension] Received request — topic: {msg.vlm_context.topic}, mastery: {msg.mastery:.0%}")

    # ── Unpack the incoming context (same for every agent) ──
    vlm_text = msg.vlm_context.raw_vlm_text or f"{msg.vlm_context.activity} — {msg.vlm_context.topic}"
    topic = msg.vlm_context.topic or "the current topic"
    mastery_pct = round(msg.mastery * 100)
    mastery_quality = msg.mastery_quality

    speech_context = ""
    if msg.vlm_context.speech_transcript:
        speech_context = f'The student just said: "{msg.vlm_context.speech_transcript}"'

    recent_obs = "\n".join(msg.recent_observations[-3:]) if msg.recent_observations else "No recent observations."

    # Extract screen details for visualization
    screen_details = vlm_text
    try:
        _data = json.loads(vlm_text) if vlm_text else {}
        screen_details = _data.get("gemini_screen_details", "") or _data.get("screen_content", "") or vlm_text
    except Exception:
        pass

    # Step 1: Pick the best tool
    tool = "visualization"  # default — prefer visuals
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

        if '"voice_call"' in tool_text:
            tool = "voice_call"
        else:
            tool = "visualization"

        logger.info(f"[Extension] Selected tool: {tool}")

    except Exception as e:
        logger.warning(f"[Extension] Tool selection failed, defaulting to visualization: {e}")
        tool = "visualization"

    # Step 2: Generate the exercise using the selected tool
    content_type = "text"
    metadata = None

    if tool == "visualization":
        # ── Full visualization pipeline via tool_visualization.py ──
        logger.info(f"[Extension] Generating visualization...")
        try:
            from agents.tools.tool_visualization import generate_visualization

            mode_to_framing = {
                "CONCEPTUAL": "conceptual",
                "APPLIED": "applied",
                "CONSOLIDATION": "extension",
            }
            framing = mode_to_framing.get(
                (msg.vlm_context.mode or "").upper(), "extension"
            )

            viz_result = generate_visualization(
                concept=topic,
                subconcept=msg.vlm_context.subtopic or "",
                confusion_hypothesis=msg.vlm_context.error_description or "",
                screen_context=screen_details,
                student_question="",
                session_id=msg.session_id,
                framing=framing,
                mastery_pct=mastery_pct,
            )

            content = viz_result.get("content", "")
            content_type = viz_result.get("content_type", "visualization")
            metadata = viz_result.get("metadata")
            tier = (metadata or {}).get("tier", "?")
            logger.info(f"[Extension] Visualization generated — tier: {tier}")

        except Exception as e:
            logger.error(f"[Extension] Visualization generation failed: {e}")
            content = "I had something cool to show you, but hit a snag. Keep going!"
            content_type = "text"
            metadata = None
    else:
        # ── voice_call ──
        try:
            system_prompt, user_template = TOOL_PROMPTS[tool]
            exercise_user_msg = user_template.format(
                vlm_context=vlm_text,
                topic=topic,
                mastery=mastery_pct,
                speech_context=speech_context,
            )

            content = _call_claude(system_prompt, exercise_user_msg, max_tokens=300)
            logger.info(f"[Extension] Generated: {content[:100]}")

        except Exception as e:
            logger.error(f"[Extension] Exercise generation failed: {e}")
            content = "I had something cool to connect this to, but hit a snag. Keep going!"

    # ── Step 3: Send response back to orchestrator (same for every agent) ──
    response = AgentResponse(
        agent_type="extension",
        content=content,
        content_type=content_type,
        tool_used=tool,
        topic=topic,
        mastery=msg.mastery,
        metadata=metadata,
    )

    await ctx.send(sender, response)
    logger.info(f"[Extension] Response sent — tool: {tool}")
