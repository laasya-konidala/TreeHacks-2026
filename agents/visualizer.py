"""
Visualizer Agent — When the conceptual agent (or orchestrator) chooses the visualization
tool, requests come here. We pass context to the visualization tool (Claude + four
options: LaTeX, D3.js, Plotly, Manim). The tool returns a UI payload; we push it to
the sidebar so it embeds and looks nice.
"""
import logging

import httpx
from uagents import Agent, Context

from agents.config import (
    VISUALIZER_SEED,
    VISUALIZER_PORT,
    AGENTVERSE_ENABLED,
    AGENTVERSE_URL,
    BACKEND_URL,
)
from agents.models import VisualizerRequest, VisualizerResponse, AgentMessage
from agents.tools.tool_visualization import generate_visualization

logger = logging.getLogger(__name__)

visualizer = Agent(
    name="math_visualizer",
    port=VISUALIZER_PORT,
    seed=VISUALIZER_SEED,
    endpoint=[f"http://127.0.0.1:{VISUALIZER_PORT}/submit"],
    agentverse=AGENTVERSE_URL if AGENTVERSE_ENABLED else None,
    mailbox=AGENTVERSE_ENABLED,
    description=(
        "Visualization agent — uses the visualization tool (LaTeX, D3.js, Plotly, Manim) "
        "to generate embeddable content for the sidebar based on student context."
    ),
    publish_agent_details=AGENTVERSE_ENABLED,
)


@visualizer.on_message(model=VisualizerRequest)
async def handle_visualization(ctx: Context, sender: str, msg: VisualizerRequest):
    """Run the visualization tool with request context and push the result to the sidebar."""
    concept = msg.concept or ""
    subconcept = getattr(msg, "subconcept", "") or ""
    confusion = getattr(msg, "confusion_hypothesis", "") or ""
    screen_context = getattr(msg, "screen_context", "") or ""
    student_question = getattr(msg, "student_question", "") or ""
    session_id = getattr(msg, "session_id", "") or ""

    logger.info(f"Visualization requested: concept={concept}, sub={subconcept}")

    # Tool does: Claude + four options → returns UI payload (tier, title, content/code/figure, narration)
    ui_payload = generate_visualization(
        concept=concept,
        subconcept=subconcept,
        confusion_hypothesis=confusion,
        screen_context=screen_context,
        student_question=student_question,
        session_id=session_id or None,
    )

    # Push to sidebar (backend broadcasts to overlay via WebSocket)
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.post(f"{BACKEND_URL}/agent-response", json=ui_payload)
            if r.status_code != 200:
                logger.warning(f"Failed to push to backend: {r.status_code} {r.text}")
    except Exception as e:
        logger.error(f"Could not POST to backend: {e}")

    # Optional: send a short structured response back to the orchestrator
    meta = ui_payload.get("metadata") or {}
    viz = meta.get("visualization") or {}
    await ctx.send(
        sender,
        VisualizerResponse(
            scene_type=viz.get("tier", "latex"),
            title=viz.get("title", ""),
            elements=[],
            animations=[],
            narration=viz.get("narration", ""),
            interactive_params=[],
            session_id=ui_payload.get("session_id", ""),
        ),
    )
