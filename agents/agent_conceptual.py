"""
Agent 1: Conceptual Understanding (Building Knowledge)

Triggered when: student is watching a video, reading notes, learning new concepts.
Goal: help them build and verify understanding of what they're consuming.

2 tools:
  - voice_call: Spoken dialogue to talk through the concept conversationally
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

# ─── Tool rotation state ───
# Tracks recent tool usage so we alternate between visualization and voice_call.
# After MAX_CONSECUTIVE uses of the same tool, force a switch.
_tool_history: list[str] = []         # most recent tools used
MAX_CONSECUTIVE_SAME_TOOL = 2         # switch after this many in a row

# ─── Agent Setup ───
CONCEPTUAL_SEED = "ambient_learning_conceptual_seed_2026"
CONCEPTUAL_PORT = 8002

_agent_kwargs = dict(
    name="conceptual_understanding",
    port=CONCEPTUAL_PORT,
    seed=CONCEPTUAL_SEED,
    description=(
        "Conceptual Understanding agent — helps students build knowledge "
        "by generating contextual voice dialogues and visualizations "
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
TOOL_SELECTION_SYSTEM = """You are a learning assistant deciding HOW to help a student who is building conceptual understanding.

You have 2 tools. Pick ONE based on the trigger reason, mastery, and what's on screen:

"voice_call" — Start a spoken dialogue with the student. Use when:
  - natural_pause + any mastery: talk through what they just saw
  - stuck: ask a simpler guiding question to unstick them
  - mode_change: bridge what they were doing to what they're doing now
  - topic_transition: talk about the connection between old and new topic
  - Low mastery: gentle check-in about the basics
  - Medium mastery: ask them to talk through their reasoning
  - High mastery: challenge them to explain an edge case or tradeoff

"visualization" — Suggest a diagram or mental model. Use when:
  - The content is abstract (equations, theory, complex relationships)
  - A visual would genuinely help more than a conversation
  - The student has been reading/watching for a while and might benefit from a different angle

Default to "voice_call" if unsure.

Respond with ONLY: {"tool": "voice_call|visualization", "reasoning": "brief reason"}"""

TOOL_SELECTION_USER = """Screen details:
{screen_details}

Topic: {topic} | Mastery: {mastery}% ({mastery_quality}) | Trigger: {trigger_reason}

Recent activity:
{recent_observations}"""


# ─── Exercise Generation Prompts ───
VOICE_CALL_SYSTEM = """You are a friendly spoken-word tutor about to start a live voice conversation with the student to explore the deep conceptual understanding of the topic at hand .
 Based on what's on their screen, generate an opening line that kicks off a short dialogue about the concept.

Adapt based on the trigger reason:
- natural_pause: "Before you move on..." — check they understood what they just saw
- topic_transition: "You just went from X to Y..." — connect the two
- stuck: Ask something SIMPLER to guide them — don't add pressure
- mode_change: Bridge theory ↔ practice — "Now that you're coding, how does X apply?"
- fallback: General check-in

Rules:
- Reference SPECIFIC things from screen_details (exact equations, code, question text, etc.)
- Do not use any novel or new concepts beyond the student's current level
- Sound natural and spoken — this will be read aloud, not displayed as text
- Open with ONE clear thought or question to get them talking
- Don't give the answer — make them think, but don't arbitrarily challenge them
- Keep it to 2-3 sentences max
- Do NOT give incorrect information, that is worse than giving no information at all"""

VOICE_CALL_USER = """Screen details:
{screen_details}

Topic: {topic} | Mastery: {mastery}% | Trigger: {trigger_reason}

Calibrate difficulty:
- Low mastery (0-30%): Start with a gentle check-in about the basics
- Medium mastery (30-70%): Ask them to talk through their reasoning
- High mastery (70-100%): Challenge them to explain an edge case or tradeoff out loud"""

VISUALIZATION_SYSTEM = """You are a learning companion. Suggest a visualization or diagram that would help the student understand what's on screen.

Adapt based on the trigger reason:
- natural_pause: visualize what they just learned as a recap
- topic_transition: show how the new topic connects visually to the old one
- stuck: simplify with a basic diagram to unstick them

Rules:
- Describe a SPECIFIC visualization tied to what's on their screen
- Keep it easily extendable to their current concept but able to adjust the specific variables
- Make it concrete: "Imagine..." or "Picture this..."
- If it's math: suggest a geometric interpretation or concrete example
- If it's code: suggest a flow diagram or state trace
- If it's theory: suggest an analogy from everyday life
- Keep it to 2-4 sentences
- Do NOT give incorrect information, that is worse than giving no information at all
- Connect the visualization to the specific content they're looking at"""

VISUALIZATION_USER = """Screen details:
{screen_details}

Topic: {topic} | Mastery: {mastery}% | Trigger: {trigger_reason}"""

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


def _extract_screen_details(raw_vlm_text: str, activity: str, topic: str) -> str:
    """
    Extract the specific screen_details from the raw VLM data.
    Falls back to activity + topic if screen_details is missing.
    """
    try:
        data = json.loads(raw_vlm_text) if raw_vlm_text else {}
        details = data.get("gemini_screen_details", "")
        if details:
            return details
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

    screen_details = _extract_screen_details(
        msg.vlm_context.raw_vlm_text,
        msg.vlm_context.activity,
        topic,
    )

    logger.info(
        f"\n{'━' * 60}\n"
        f"  🧐 CONCEPTUAL AGENT — Request received\n"
        f"  Topic:   {topic} | Mastery: {mastery_pct}% ({mastery_quality})\n"
        f"  Trigger: {trigger_reason}\n"
        f"  Screen:  {screen_details[:150]}\n"
        f"{'━' * 60}"
    )

    recent_obs = "\n".join(msg.recent_observations[-3:]) if msg.recent_observations else "No recent observations."

    # Step 1: Pick the best tool (LLM chooses, but force switch after 2 in a row)
    global _tool_history
    consecutive_same = (
        len(_tool_history) >= MAX_CONSECUTIVE_SAME_TOOL
        and all(t == _tool_history[-1] for t in _tool_history[-MAX_CONSECUTIVE_SAME_TOOL:])
    )

    if consecutive_same:
        # Force switch — used the same tool too many times
        last = _tool_history[-1]
        tool = "voice_call" if last == "visualization" else "visualization"
        logger.info(f"  🔀 Step 1: Forced switch → {tool} (used {last} {MAX_CONSECUTIVE_SAME_TOOL}x in a row)")
    else:
        # Let Claude decide based on context
        logger.info("  🔀 Step 1: Picking tool (LLM call 1)...")
        tool = "voice_call"  # default
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
                tool = "voice_call"
            logger.info(f"  🔀 Tool selected: {tool}  (Claude said: {tool_text[:80]})")
        except Exception as e:
            logger.warning(f"  ⚠️ Tool selection failed, defaulting to voice_call: {e}")
            tool = "voice_call"

    _tool_history.append(tool)

    # Step 2: Generate the exercise using the selected tool
    content_type = "text"
    metadata = None

    if tool == "visualization":
        # ── Full visualization pipeline via tool_visualization.py ──
        logger.info("  🎨 Step 2: Generating visualization (LLM call 2)...")
        try:
            from agents.tools.tool_visualization import generate_visualization

            # Derive framing from VLM mode (not hardcoded)
            mode_to_framing = {
                "CONCEPTUAL": "conceptual",
                "APPLIED": "applied",
                "CONSOLIDATION": "extension",
            }
            framing = mode_to_framing.get(
                (msg.vlm_context.mode or "").upper(), "conceptual"
            )
            logger.info(f"  🎨 Framing: {framing} (from VLM mode: {msg.vlm_context.mode})")

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
            viz_obj = (metadata or {}).get("visualization") or {}
            viz_title = viz_obj.get("title", "?")
            logger.info(f"  🎨 Visualization generated — tier: {tier}, title: {viz_title}")

            # ── Debug: print full visualization payload ──
            viz_code = viz_obj.get("code") or viz_obj.get("content") or ""
            logger.info(f"\n{'─' * 60}")
            logger.info(f"  🎨 FULL VISUALIZATION OUTPUT")
            logger.info(f"  Tier: {tier} | Title: {viz_title}")
            logger.info(f"  Narration: {viz_obj.get('narration', '')}")
            if tier == "d3":
                logger.info(f"  D3 Code:\n{viz_code}")
            elif tier == "latex":
                logger.info(f"  LaTeX: {viz_code}")
            elif tier == "plotly":
                logger.info(f"  Plotly figure: {json.dumps(viz_obj.get('plotly_figure', {}), indent=2)[:500]}")
            elif tier == "manim":
                logger.info(f"  Manim Code:\n{viz_code}")
            logger.info(f"{'─' * 60}")

        except Exception as e:
            logger.error(f"  ❌ Visualization generation failed: {e}")
            content = "I wanted to show you a visualization, but ran into an issue. Keep going!"
            content_type = "text"
            metadata = None
    else:
        # ── voice_call or other text-based tools ──
        logger.info(f"  🗣️ Step 2: Generating {tool} content (LLM call 2)...")
        try:
            system_prompt, user_template = TOOL_PROMPTS[tool]
            exercise_user_msg = user_template.format(
                screen_details=screen_details,
                topic=topic,
                mastery=mastery_pct,
                trigger_reason=trigger_reason,
            )

            content = _call_claude(system_prompt, exercise_user_msg, max_tokens=300)
            logger.info(f"  🗣️ Generated: {content[:120]}")

        except Exception as e:
            logger.error(f"  ❌ Exercise generation failed: {e}")
            content = "Hmm, I wanted to ask you something about what you're reading, but I ran into an issue. Keep going!"

    # Step 3: Send response back to orchestrator
    response = AgentResponse(
        agent_type="conceptual",
        content=content,
        content_type=content_type,
        tool_used=tool,
        topic=topic,
        mastery=msg.mastery,
        metadata=metadata,
    )

    await ctx.send(sender, response)
    logger.info(
        f"  📤 Response sent → orchestrator\n"
        f"     tool: {tool} | content_type: {content_type} | trigger: {trigger_reason}\n"
        f"{'━' * 60}"
    )
