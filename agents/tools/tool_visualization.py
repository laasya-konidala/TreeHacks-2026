"""
Tool: Visualization Suggestions

Generates text-based visualization prompts that help the student
build mental models.
"""
import logging
import anthropic

from agents.config import ANTHROPIC_API_KEY, CLAUDE_MODEL

logger = logging.getLogger(__name__)

_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


async def suggest_visualization(
    vlm_context: str,
    topic: str,
    mastery_pct: int,
    speech_context: str = "",
) -> str:
    """Suggest a mental visualization or diagram for the current concept."""
    system = """You are a learning companion. Suggest a quick mental visualization 
or diagram that would help the student understand what's on screen.

Rules:
- Describe a SPECIFIC visualization related to what's on screen
- Make it concrete: "Imagine..." or "Picture this..."
- Connect the visualization to the specific content they're looking at
- If it's math/code, suggest a concrete example or analogy
- 2-4 sentences max"""

    user_msg = f"""What's on their screen right now:
{vlm_context}

Their mastery of "{topic}" is {mastery_pct}%.

{speech_context}"""

    try:
        response = _client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=300,
            system=system,
            messages=[{"role": "user", "content": user_msg}],
        )
        return response.content[0].text
    except Exception as e:
        logger.error(f"[tool_visualization] Error: {e}")
        return "Visualization suggestion failed — try again soon."
