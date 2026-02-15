"""
Visualization tool: agent chooses this tool → we pass context to Claude with four
options (LaTeX, D3.js, Plotly, Manim). Claude returns code/content; we embed it
into the sidebar UI and make it look nice.

Tiers:
- latex: Definitions, single equations, step-by-step math. Rendered with KaTeX.
- d3: Custom diagrams, networks, flowcharts, node-link. Claude returns JS; we run in sandbox.
- plotly: 2D/3D charts, scatter, regression, surfaces. Claude returns figure JSON.
- manim: Narrative animations, 3B1B-style deep dives. Claude returns Manim script (we run via backend or show placeholder).
"""
import json
import logging
import re
from typing import Any, Optional

import anthropic

from agents.config import ANTHROPIC_API_KEY, CLAUDE_MODEL

logger = logging.getLogger(__name__)

_client: Optional[anthropic.Anthropic] = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    return _client


# ─── When to use each tier (for Claude) ───
VISUALIZATION_SYSTEM = """You are a learning companion. The student's agent has chosen the "visualization" tool. Given the current context (what's on screen, what they're learning, any confusion), you must:

1. Choose ONE output type: latex | d3 | plotly | manim
2. Return valid, runnable code/content that the UI will embed in the sidebar.

## When to use each type

**latex**
- Use for: single equations, definitions, identities, step-by-step math, gradient/derivative notation, matrix formulas, short proofs.
- Examples: "What is the gradient?", "Write the chain rule", "Show the least-squares normal equation", "Definition of eigenvalue", "Backprop equation".
- Do NOT use for: flowcharts, graphs with data, interactive sliders, animations over time.

**d3**
- Use for: flowcharts, process diagrams, node-link graphs, trees, state machines, custom SVG diagrams, network layouts, force-directed graphs, anything that needs custom shapes/arrows/labels and is more diagram than chart.
- Examples: "Flow of gradient descent (steps)", "Neural network layer diagram", "Decision tree", "Graph of operations", "Data pipeline flowchart".
- Do NOT use for: raw math (use latex) or numeric charts (use plotly).

**plotly**
- Use for: 2D/3D data plots, scatter plots, line charts, regression lines, surfaces, histograms, anything with axes and numeric data or functions f(x).
- Examples: "Plot loss curve", "Scatter plot with regression line", "Gradient descent path on a surface", "Distribution of values", "Function y = f(x)".
- Do NOT use for: pure equations (latex) or abstract diagrams (d3).

**manim**
- Use for: multi-step narrative animations, 3Blue1Brown-style "story" explanations, transformations over time, when you want the student to watch an animation that builds up.
- Examples: "Animate how eigenvectors scale", "Show matrix multiplication as transformation", "Build up the integral step by step", "Animate gradient descent step-by-step".
- Do NOT use for: static equations (latex), static diagrams (d3), or static charts (plotly). Prefer manim only when animation adds real pedagogical value.

## Output format

Respond with a single JSON object (no markdown, no extra text). Use this exact shape:

For **latex**:
{
  "tier": "latex",
  "title": "Short title for the card",
  "narration": "1-2 sentences explaining what to notice (for the sidebar caption).",
  "content": "\\\\nabla f(x) = \\\\left( \\\\frac{\\\\partial f}{\\\\partial x_1}, ... \\\\right)"
}

For **d3**:
{
  "tier": "d3",
  "title": "Short title",
  "narration": "1-2 sentences for the caption.",
  "code": "// JavaScript that receives a container DOM element and draws into it. Use D3 or vanilla SVG.\\nfunction draw(container) { ... }"
}

For **plotly**:
{
  "tier": "plotly",
  "title": "Short title",
  "narration": "1-2 sentences.",
  "figure": {
    "data": [ { "x": [...], "y": [...], "type": "scatter", "mode": "markers" }, ... ],
    "layout": { "title": "...", "xaxis": { "title": "x" }, "yaxis": { "title": "y" }, "margin": { "t": 40, "b": 40, "l": 50, "r": 20 }, "paper_bgcolor": "rgba(0,0,0,0)", "plot_bgcolor": "rgba(255,255,255,0.05)", "font": { "color": "#e5e7eb", "size": 12 } }
  }
}

For **manim**:
{
  "tier": "manim",
  "title": "Short title",
  "narration": "What the animation will show.",
  "code": "from manim import *\\n\\nclass ConceptScene(Scene):\\n    def construct(self):\\n        ..."
}

Rules:
- Escape backslashes in JSON (e.g. \\\\nabla for LaTeX).
- For plotly, "data" and "layout" must be valid Plotly.js figure spec.
- For d3, "code" must be a single function draw(container) or equivalent that the UI can run; use only standard browser APIs and D3 if needed.
- For manim, "code" must be a valid Manim Community Edition script with a Scene class.
- Always include "title" and "narration" for the UI card.
"""


# ─── Agent framing: how to slant the visualization based on which agent called ───
FRAMING = {
    "conceptual": """The student is LEARNING this concept (watching a video, reading notes).
Goal: help them BUILD UNDERSTANDING.
- Visualize what the concept MEANS, not how to use it
- Show the intuition, the "why", the geometric/physical interpretation
- E.g., for eigenvalues: show how a matrix transforms vectors, some stretch (eigenvectors) and some don't
- Keep it simple if mastery is low — one core idea visualized clearly
- Prioritize clarity over complexity""",

    "applied": """The student is SOLVING A PROBLEM (coding, doing exercises, working through steps).
Goal: help them GET UNSTUCK or CHECK THEIR WORK.
- Visualize the SPECIFIC problem they're working on, using the EXACT values from their screen
- Show intermediate steps, where they might have gone wrong, or what the answer should look like
- E.g., for eigenvalues: plot the characteristic polynomial with THEIR matrix values, show where the roots are
- Be concrete and practical, not abstract
- Use the exact numbers, variables, and expressions from the screen""",

    "extension": """The student has decent mastery and we want to PUSH DEEPER.
Goal: show connections, generalizations, or "what if" explorations.
- Visualize what happens when you change parameters, edge cases, connections to other topics
- E.g., for eigenvalues: interactive sliders to change matrix entries and watch eigenvectors rotate/scale in real-time
- Make it exploratory — let them discover something, don't just show them
- Prioritize interactivity (sliders, clickable elements, hover effects) over static content
- Show the bigger picture""",
}


def _build_user_message(
    concept: str,
    subconcept: str,
    confusion_hypothesis: str,
    screen_context: str,
    student_question: str,
    framing: str = "conceptual",
    mastery_pct: int = 0,
) -> str:
    framing_text = FRAMING.get(framing, FRAMING["conceptual"])

    return f"""Current context for the visualization:

Concept: {concept or "general"}
Subconcept: {subconcept or "—"}
Student mastery: {mastery_pct}% — {"beginner, keep it simple" if mastery_pct < 30 else "intermediate, moderate complexity" if mastery_pct < 70 else "advanced, can handle depth and interactivity"}
What's on their screen: {screen_context or "—"}
What we suspect they need help with: {confusion_hypothesis or "—"}
Student question (if any): {student_question or "—"}

Agent framing (how to slant this visualization):
{framing_text}

Choose latex, d3, plotly, or manim and return ONLY the JSON object (no markdown, no explanation)."""


def _parse_json_from_response(text: str) -> Optional[dict]:
    """Extract a JSON object from Claude's response (may be inside markdown code block)."""
    text = (text or "").strip()
    # Strip markdown code block if present
    if "```" in text:
        match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if match:
            text = match.group(1).strip()
    # Find first { ... }
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    pass
                break
    return None


def generate_visualization(
    concept: str = "",
    subconcept: str = "",
    confusion_hypothesis: str = "",
    screen_context: str = "",
    student_question: str = "",
    session_id: Optional[str] = None,
    framing: str = "conceptual",
    mastery_pct: int = 0,
) -> dict[str, Any]:
    """
    Call Claude with context and the four options (latex, d3, plotly, manim).
    Claude returns code/content; we normalize it into the UI payload the sidebar
    expects so it can embed and display it nicely.

    Args:
        framing: "conceptual" | "applied" | "extension" — how to slant the visualization
        mastery_pct: 0-100 student mastery level — calibrates complexity

    Returns a dict suitable for the overlay: content_type, content, metadata.tier,
    metadata.visualization with tier-specific fields (content, code, figure, etc.).
    """
    session_id = session_id or ""
    user_msg = _build_user_message(
        concept, subconcept, confusion_hypothesis, screen_context, student_question,
        framing=framing, mastery_pct=mastery_pct,
    )

    try:
        response = _get_client().messages.create(
            model=CLAUDE_MODEL,
            max_tokens=2048,
            system=VISUALIZATION_SYSTEM,
            messages=[{"role": "user", "content": user_msg}],
        )
        text = response.content[0].text if response.content else ""
    except Exception as e:
        logger.error(f"[tool_visualization] Claude API error: {e}")
        return _fallback_ui_payload("latex", concept, session_id, error=str(e))

    parsed = _parse_json_from_response(text)
    if not parsed or "tier" not in parsed:
        logger.warning("[tool_visualization] Could not parse JSON from Claude; using fallback")
        return _fallback_ui_payload("latex", concept, session_id)

    tier = (parsed.get("tier") or "latex").lower()
    if tier not in ("latex", "d3", "plotly", "manim"):
        tier = "latex"

    title = parsed.get("title") or f"Visualizing {concept or 'concept'}"
    narration = parsed.get("narration") or ""

    # Build UI payload: same shape the overlay expects (metadata.tier, metadata.visualization)
    visualization: dict[str, Any] = {
        "tier": tier,
        "title": title,
        "narration": narration,
    }

    if tier == "latex":
        visualization["format"] = "latex"
        visualization["content"] = parsed.get("content") or "\\text{No content generated.}"
    elif tier == "d3":
        visualization["code"] = parsed.get("code") or "function draw(container) { container.textContent = 'No diagram generated.'; }"
    elif tier == "plotly":
        fig = parsed.get("figure")
        if isinstance(fig, dict) and ("data" in fig or "layout" in fig):
            visualization["plotly_figure"] = fig
        else:
            visualization["plotly_figure"] = {
                "data": [{"x": [0, 1, 2], "y": [0, 1, 2], "type": "scatter", "mode": "lines"}],
                "layout": {"title": title, "margin": {"t": 40, "b": 40, "l": 50, "r": 20}},
            }
    elif tier == "manim":
        visualization["code"] = parsed.get("code") or "# Manim script not generated"
        # Optional: backend can run manim and set status_url; here we just pass code for placeholder or future use
        visualization["status_url"] = None

    return {
        "content_type": "visualization",
        "content": narration,
        "agent_type": "visualizer",
        "session_id": session_id,
        "tool_used": "visualization",
        "metadata": {
            "tier": tier,
            "concept": concept or "",
            "visualization": visualization,
            "scene": visualization,
        },
    }


def _fallback_ui_payload(
    tier: str,
    concept: str,
    session_id: str,
    error: Optional[str] = None,
) -> dict[str, Any]:
    """Fallback when Claude fails or returns invalid JSON."""
    title = f"Visualizing {concept or 'concept'}"
    narration = error or "Could not generate visualization. Try again."
    visualization: dict[str, Any] = {
        "tier": tier,
        "title": title,
        "narration": narration,
        "format": "latex",
        "content": "\\text{" + (error or "No content") + "}",
    }
    return {
        "content_type": "visualization",
        "content": narration,
        "agent_type": "visualizer",
        "session_id": session_id,
        "tool_used": "visualization",
        "metadata": {
            "tier": tier,
            "concept": concept or "",
            "visualization": visualization,
            "scene": visualization,
        },
    }


async def suggest_visualization(
    vlm_context: str,
    topic: str,
    mastery_pct: int,
    speech_context: str = "",
) -> str:
    """Lightweight text-only suggestion (no code). Use when you only need a prose hint."""
    # Reuse a minimal prompt for text-only
    user_msg = f"""What's on their screen: {vlm_context}
Topic: {topic}, mastery ~{mastery_pct}%.
{speech_context}

Suggest a quick mental visualization in 2-3 sentences ("Imagine...", "Picture this..."). No code, no JSON."""
    try:
        response = _get_client().messages.create(
            model=CLAUDE_MODEL,
            max_tokens=200,
            messages=[{"role": "user", "content": user_msg}],
        )
        return response.content[0].text or "Visualization suggestion failed."
    except Exception as e:
        logger.error(f"[tool_visualization] suggest_visualization: {e}")
        return "Visualization suggestion failed — try again soon."
