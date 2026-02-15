"""
Visualizer Agent — Generates structured scene descriptions for 3B1B-style visualizations.
The Electron frontend renders these as interactive visualizations.
"""
import json
import logging
from typing import Optional

from google import genai
from google.genai import types
from uagents import Agent, Context

from agents.config import (
    VISUALIZER_SEED, VISUALIZER_PORT, GEMINI_MODEL, GEMINI_API_KEY,
    AGENTVERSE_ENABLED, AGENTVERSE_URL,
)
from agents.models import VisualizerRequest, VisualizerResponse, AgentMessage

logger = logging.getLogger(__name__)

# ─── Agent Setup ───
visualizer = Agent(
    name="math_visualizer",
    port=VISUALIZER_PORT,
    seed=VISUALIZER_SEED,
    endpoint=[f"http://127.0.0.1:{VISUALIZER_PORT}/submit"],
    agentverse=AGENTVERSE_URL if AGENTVERSE_ENABLED else None,
    mailbox=AGENTVERSE_ENABLED,
    description=(
        "Mathematical visualization agent — generates 3Blue1Brown-style "
        "interactive visual explanations with scene descriptions, animations, "
        "and adjustable parameters for concepts in ML, calculus, and linear algebra."
    ),
    publish_agent_details=AGENTVERSE_ENABLED,
)

gemini_client = genai.Client(api_key=GEMINI_API_KEY)

# ─── Template Library ───
TEMPLATES = {
    "gradient_descent": {
        "type": "interactive_graph",
        "title": "Gradient Descent — Finding the Minimum",
        "elements": [
            {"type": "curve", "fn": "x^2 + 2*sin(x)", "color": "#3b82f6", "label": "Loss function"},
            {"type": "point", "x": 3.0, "y": None, "color": "#ef4444", "label": "Current position", "animate": True},
            {"type": "arrow", "from_point": "current", "direction": "gradient", "color": "#22c55e", "label": "Gradient"},
            {"type": "trail", "color": "#ef444480", "label": "Path taken"},
        ],
        "animations": [
            {"step": 1, "action": "move_point", "target": "gradient_step", "duration": 500},
            {"step": 2, "action": "draw_trail", "duration": 200},
            {"step": 3, "action": "update_arrow", "duration": 300},
        ],
        "narration": "Watch how the point follows the negative gradient downhill. Each step is proportional to the slope — steep slopes mean bigger steps. Notice how it slows down near the minimum where the gradient approaches zero.",
        "interactive_params": [
            {"name": "learning_rate", "type": "slider", "min": 0.01, "max": 2.0, "default": 0.1, "label": "Learning Rate"},
            {"name": "start_x", "type": "slider", "min": -5, "max": 5, "default": 3, "label": "Starting Point"},
        ],
    },
    "linear_regression": {
        "type": "interactive_graph",
        "title": "Linear Regression — Fitting a Line",
        "elements": [
            {"type": "scatter", "data": "generated", "color": "#3b82f6", "label": "Data points"},
            {"type": "line", "slope": 1, "intercept": 0, "color": "#ef4444", "label": "Regression line"},
            {"type": "residuals", "color": "#22c55e80", "label": "Residuals (errors)"},
        ],
        "animations": [
            {"step": 1, "action": "show_residuals", "duration": 500},
            {"step": 2, "action": "rotate_line", "target": "minimize_residuals", "duration": 2000},
        ],
        "narration": "The green lines show the residuals — the distance between each point and the line. Linear regression finds the line that minimizes the sum of squared residuals. Watch how the line rotates to find the best fit.",
        "interactive_params": [
            {"name": "slope", "type": "slider", "min": -3, "max": 3, "default": 1, "label": "Slope"},
            {"name": "intercept", "type": "slider", "min": -5, "max": 5, "default": 0, "label": "Intercept"},
            {"name": "noise", "type": "slider", "min": 0, "max": 3, "default": 1, "label": "Noise Level"},
        ],
    },
    "neural_network": {
        "type": "comparison",
        "title": "Neural Network — Forward Pass",
        "elements": [
            {"type": "network", "layers": [3, 4, 4, 2], "color_scheme": "activation_magnitude"},
            {"type": "weights", "show": True, "color_by": "sign"},
            {"type": "activations", "show": True, "color_by": "magnitude"},
        ],
        "animations": [
            {"step": 1, "action": "forward_pass", "layer_delay": 400, "duration": 2000},
            {"step": 2, "action": "highlight_path", "path": "max_activation", "duration": 1000},
        ],
        "narration": "Watch the signal flow through the network. Each neuron computes a weighted sum of its inputs, then applies an activation function. The brightness shows the activation magnitude — brighter means stronger signal.",
        "interactive_params": [
            {"name": "input_values", "type": "vector", "size": 3, "label": "Input Values"},
            {"name": "activation", "type": "select", "options": ["relu", "sigmoid", "tanh"], "default": "relu", "label": "Activation"},
        ],
    },
    "probability_distribution": {
        "type": "interactive_graph",
        "title": "Probability Distributions",
        "elements": [
            {"type": "curve", "fn": "normal", "color": "#3b82f6", "label": "Distribution", "fill": True},
            {"type": "vline", "x": 0, "color": "#ef4444", "label": "Mean"},
            {"type": "region", "from": -1, "to": 1, "color": "#22c55e40", "label": "1 std dev (68%)"},
        ],
        "animations": [
            {"step": 1, "action": "morph_distribution", "duration": 1000},
        ],
        "narration": "The bell curve shows how probability is distributed. About 68% of values fall within one standard deviation of the mean. Watch how changing the parameters reshapes the distribution.",
        "interactive_params": [
            {"name": "mean", "type": "slider", "min": -5, "max": 5, "default": 0, "label": "Mean (μ)"},
            {"name": "std", "type": "slider", "min": 0.1, "max": 3, "default": 1, "label": "Std Dev (σ)"},
        ],
    },
    "loss_landscape": {
        "type": "3d_surface",
        "title": "Loss Landscape — 3D View",
        "elements": [
            {"type": "surface", "fn": "loss_function", "colormap": "viridis"},
            {"type": "path_3d", "color": "#ef4444", "label": "Optimization path"},
            {"type": "minimum", "color": "#22c55e", "label": "Global minimum"},
        ],
        "animations": [
            {"step": 1, "action": "rotate_view", "duration": 3000},
            {"step": 2, "action": "trace_path", "duration": 2000},
        ],
        "narration": "This is the loss landscape in 3D. The height represents the loss value. Gradient descent follows the surface downhill. Notice the valleys and ridges — getting stuck on a ridge or in a local minimum is a real challenge.",
        "interactive_params": [
            {"name": "view_angle", "type": "slider", "min": 0, "max": 360, "default": 45, "label": "View Angle"},
            {"name": "learning_rate", "type": "slider", "min": 0.01, "max": 1.0, "default": 0.1, "label": "Learning Rate"},
        ],
    },
}

# Concept-to-template mapping
CONCEPT_TEMPLATE_MAP = {
    "gradient": "gradient_descent",
    "gradient_descent": "gradient_descent",
    "optimization": "gradient_descent",
    "linear_regression": "linear_regression",
    "regression": "linear_regression",
    "least_squares": "linear_regression",
    "neural_network": "neural_network",
    "neural_net": "neural_network",
    "forward_pass": "neural_network",
    "backpropagation": "neural_network",
    "probability": "probability_distribution",
    "distribution": "probability_distribution",
    "normal": "probability_distribution",
    "gaussian": "probability_distribution",
    "loss": "loss_landscape",
    "loss_function": "loss_landscape",
    "loss_landscape": "loss_landscape",
}


def _find_template(concept: str) -> Optional[dict]:
    """Find a matching visualization template."""
    concept_lower = concept.lower().replace(" ", "_")

    # Direct match
    if concept_lower in TEMPLATES:
        return TEMPLATES[concept_lower]

    # Keyword match
    for keyword, template_id in CONCEPT_TEMPLATE_MAP.items():
        if keyword in concept_lower:
            return TEMPLATES[template_id]

    return None


def _generate_novel_visualization(concept: str, subconcept: str, confusion: str) -> dict:
    """Generate a novel visualization via Gemini API."""
    prompt = (
        f"Create a visualization description for teaching: {concept} (specifically: {subconcept}).\n"
        f"The student is confused about: {confusion}\n\n"
        "Design a 3Blue1Brown-style interactive visualization. Return ONLY valid JSON (no markdown):\n"
        "{\n"
        '  "type": "interactive_graph" | "3d_surface" | "comparison" | "animation_sequence",\n'
        '  "title": "descriptive title",\n'
        '  "elements": [{"type": "...", ...}],\n'
        '  "animations": [{"step": 1, "action": "...", "duration": 500}],\n'
        '  "narration": "3B1B-style insight text explaining what to notice",\n'
        '  "interactive_params": [{"name": "...", "type": "slider", "min": 0, "max": 1, "default": 0.5, "label": "..."}]\n'
        "}"
    )

    try:
        response = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction="You create mathematical visualization descriptions. Be specific about geometry, colors, and animations. Think like 3Blue1Brown.",
                max_output_tokens=500,
            ),
        )
        text = (response.text or "").replace("```json", "").replace("```", "").strip()
        return json.loads(text)
    except Exception as e:
        logger.error(f"Novel visualization generation failed: {e}")
        # Fallback to generic template
        return TEMPLATES.get("gradient_descent", {
            "type": "interactive_graph",
            "title": f"Visualizing {concept}",
            "elements": [],
            "animations": [],
            "narration": f"Let's visualize {concept} to build intuition.",
            "interactive_params": [],
        })


@visualizer.on_message(model=VisualizerRequest)
async def handle_visualization(ctx: Context, sender: str, msg: VisualizerRequest):
    """Generate a visualization scene description."""
    logger.info(f"Visualization requested: concept={msg.concept}, sub={msg.subconcept}")

    # Try template library first
    template = _find_template(msg.concept)

    if template:
        scene = template
    else:
        scene = _generate_novel_visualization(
            msg.concept, msg.subconcept, msg.confusion_hypothesis,
        )

    # Send structured visualization response
    await ctx.send(
        sender,
        VisualizerResponse(
            scene_type=scene.get("type", "interactive_graph"),
            title=scene.get("title", f"Visualizing {msg.concept}"),
            elements=scene.get("elements", []),
            animations=scene.get("animations", []),
            narration=scene.get("narration", ""),
            interactive_params=scene.get("interactive_params", []),
            session_id=msg.session_id,
        ),
    )

    # Also send as AgentMessage for UI
    await ctx.send(
        sender,
        AgentMessage(
            content=scene.get("narration", ""),
            content_type="visualization",
            agent_type="visualizer",
            session_id=msg.session_id,
            metadata={
                "scene": scene,
                "concept": msg.concept,
            },
        ),
    )
