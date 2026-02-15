"""
Assessor Agent — Contrastive assessment for testing and extending understanding.
Generates "what if you changed ONE thing?" challenges.
"""
import json
import logging
from typing import Optional

from google import genai
from google.genai import types
from uagents import Agent, Context

from agents.config import (
    ASSESSOR_SEED, ASSESSOR_PORT, GEMINI_MODEL, GEMINI_API_KEY,
    AGENTVERSE_ENABLED, AGENTVERSE_URL,
)
from agents.models import AssessorRequest, AssessorResponse, AgentMessage

logger = logging.getLogger(__name__)

# ─── Agent Setup ───
assessor = Agent(
    name="contrastive_assessor",
    port=ASSESSOR_PORT,
    seed=ASSESSOR_SEED,
    endpoint=[f"http://127.0.0.1:{ASSESSOR_PORT}/submit"],
    agentverse=AGENTVERSE_URL if AGENTVERSE_ENABLED else None,
    mailbox=AGENTVERSE_ENABLED,
    description=(
        "Contrastive assessment agent — generates 'what if you changed ONE "
        "thing?' challenges to test and extend understanding. Includes a "
        "pattern library for ML/stats concepts and generates novel contrasts."
    ),
    publish_agent_details=AGENTVERSE_ENABLED,
)

gemini_client = genai.Client(api_key=GEMINI_API_KEY)

# ─── Contrastive Pattern Library ───
PATTERNS = {
    "linear_regression_bias": {
        "challenge": "What would happen if you removed the bias term (intercept) from your linear regression?",
        "what_changes": "The regression line is forced through the origin. Unless the data truly passes through (0,0), this introduces systematic error and the model underfits.",
        "expected_insight": "The bias term allows the model to fit data that doesn't pass through the origin — it's a degree of freedom that reduces error.",
        "connects_to": "regularization",
    },
    "l1_vs_l2": {
        "challenge": "You're using L2 regularization. What changes if you switch to L1 (Lasso) instead?",
        "what_changes": "L1 tends to produce sparse solutions — some weights go exactly to zero. L2 shrinks all weights but rarely zeroes them out.",
        "expected_insight": "L1 does implicit feature selection. L2 distributes the penalty more evenly. The geometry (diamond vs circle constraint) explains why.",
        "connects_to": "feature_selection",
    },
    "learning_rate_too_high": {
        "challenge": "What happens if you multiply your learning rate by 10?",
        "what_changes": "The optimization likely diverges or oscillates wildly around the minimum instead of converging to it.",
        "expected_insight": "Learning rate controls the step size in gradient descent. Too large and you overshoot; too small and you converge too slowly.",
        "connects_to": "gradient_descent_convergence",
    },
    "gradient_descent_vs_closed_form": {
        "challenge": "Instead of gradient descent, what if you used the normal equation (closed-form solution)?",
        "what_changes": "You get the exact solution in one step, but it requires inverting a matrix — O(n³) complexity. For large n, gradient descent is faster.",
        "expected_insight": "There's a tradeoff: exact vs iterative. Gradient descent scales better but needs tuning (learning rate, iterations).",
        "connects_to": "computational_complexity",
    },
    "relu_vs_sigmoid": {
        "challenge": "You're using ReLU activation. What changes if you swap to sigmoid?",
        "what_changes": "Sigmoid squashes to (0,1) and suffers from vanishing gradients in deep networks. ReLU is unbounded above and has constant gradient for positive inputs.",
        "expected_insight": "ReLU solved the vanishing gradient problem for deep networks, but can cause 'dead neurons'. The choice of activation affects trainability.",
        "connects_to": "vanishing_gradients",
    },
    "batch_size": {
        "challenge": "What if you changed your batch size from 32 to 1 (pure SGD)?",
        "what_changes": "Each update is noisier but more frequent. You get a noisy approximation of the gradient. Training is less stable but can escape local minima.",
        "expected_insight": "Batch size controls the variance-computation tradeoff. Larger batches = smoother gradients but more computation per update.",
        "connects_to": "stochastic_optimization",
    },
    "dropout": {
        "challenge": "What if you removed all dropout layers from your network?",
        "what_changes": "The network is more likely to overfit — it can memorize training data without the regularizing effect of randomly zeroing activations.",
        "expected_insight": "Dropout forces the network to be robust by not relying on any single neuron. It's like training an ensemble of sub-networks.",
        "connects_to": "regularization",
    },
    "softmax_temperature": {
        "challenge": "What if you divided the logits by 0.1 before softmax (low temperature)?",
        "what_changes": "The probability distribution becomes much 'sharper' — the highest logit gets nearly all the probability mass.",
        "expected_insight": "Temperature controls the entropy of the output distribution. Low temp = confident/greedy, high temp = exploratory/uniform.",
        "connects_to": "probability_distributions",
    },
}

# Keyword mapping to patterns
CONCEPT_PATTERN_MAP = {
    "linear_regression": ["linear_regression_bias"],
    "regularization": ["l1_vs_l2"],
    "l1": ["l1_vs_l2"],
    "l2": ["l1_vs_l2"],
    "lasso": ["l1_vs_l2"],
    "ridge": ["l1_vs_l2"],
    "learning_rate": ["learning_rate_too_high"],
    "gradient_descent": ["gradient_descent_vs_closed_form", "learning_rate_too_high"],
    "normal_equation": ["gradient_descent_vs_closed_form"],
    "relu": ["relu_vs_sigmoid"],
    "sigmoid": ["relu_vs_sigmoid"],
    "activation": ["relu_vs_sigmoid"],
    "batch": ["batch_size"],
    "sgd": ["batch_size"],
    "dropout": ["dropout"],
    "softmax": ["softmax_temperature"],
    "temperature": ["softmax_temperature"],
}


def _find_pattern(concept: str) -> Optional[dict]:
    """Find a matching contrastive pattern for a concept."""
    concept_lower = concept.lower().replace(" ", "_")

    # Direct match
    if concept_lower in PATTERNS:
        return PATTERNS[concept_lower]

    # Keyword match
    for keyword, pattern_ids in CONCEPT_PATTERN_MAP.items():
        if keyword in concept_lower:
            return PATTERNS[pattern_ids[0]]

    return None


def _generate_novel_contrast(concept: str, user_solution: str) -> dict:
    """Generate a novel contrastive example via Gemini API."""
    prompt = (
        f"Student just worked on {concept}: {user_solution[:500]}\n\n"
        "Generate ONE contrastive example that changes ONE thing to test their "
        "understanding. The contrast should reveal a key insight about the concept.\n\n"
        "Return ONLY valid JSON (no markdown fences):\n"
        '{"challenge": "question to present", "what_changes": "what is different and why", '
        '"expected_insight": "what they should realize", "connects_to": "related concept"}'
    )

    try:
        response = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction="You are a master educator who creates contrastive examples to test understanding. Be specific and insightful.",
                max_output_tokens=300,
            ),
        )
        text = (response.text or "").replace("```json", "").replace("```", "").strip()
        return json.loads(text)
    except Exception as e:
        logger.error(f"Novel contrast generation failed: {e}")
        return {
            "challenge": f"What would change about your approach to {concept} if you had to explain it to a beginner?",
            "what_changes": "Simplifying forces you to identify the core concept vs implementation details.",
            "expected_insight": "Understanding = ability to simplify without losing the essential idea.",
            "connects_to": "foundational_understanding",
        }


@assessor.on_message(model=AssessorRequest)
async def handle_assessment(ctx: Context, sender: str, msg: AssessorRequest):
    """Generate a contrastive challenge for the student."""
    logger.info(f"Assessment requested: concept={msg.concept}, mastery={msg.mastery_level}")

    # Try pattern library first
    pattern = _find_pattern(msg.concept)

    if pattern:
        challenge = pattern["challenge"]
        what_changes = pattern["what_changes"]
        expected_insight = pattern["expected_insight"]
        connects_to = pattern["connects_to"]
    else:
        # Generate novel contrast via Gemini
        novel = _generate_novel_contrast(msg.concept, msg.user_solution)
        challenge = novel["challenge"]
        what_changes = novel["what_changes"]
        expected_insight = novel["expected_insight"]
        connects_to = novel["connects_to"]

    # Send assessment response back to orchestrator
    await ctx.send(
        sender,
        AssessorResponse(
            challenge=challenge,
            what_changes=what_changes,
            expected_insight=expected_insight,
            connects_to=connects_to,
            session_id=msg.session_id,
        ),
    )

    # Also send as AgentMessage for UI display
    await ctx.send(
        sender,
        AgentMessage(
            content=challenge,
            content_type="challenge",
            agent_type="assessor",
            session_id=msg.session_id,
            metadata={
                "what_changes": what_changes,
                "expected_insight": expected_insight,
                "connects_to": connects_to,
                "concept": msg.concept,
            },
        ),
    )
