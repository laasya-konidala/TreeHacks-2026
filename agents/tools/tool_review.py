"""
Tool: Review / Connect to Prior Knowledge

Briefly connects what the student is currently learning to foundational
concepts or previously covered material.
"""
import logging
import anthropic

from agents.config import ANTHROPIC_API_KEY, CLAUDE_MODEL

logger = logging.getLogger(__name__)

_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


async def connect_to_prior(
    vlm_context: str,
    topic: str,
    mastery_pct: int,
    speech_context: str = "",
) -> str:
    """Make a quick connection between current learning and prior knowledge."""
    system = """You are a learning companion. Briefly connect what the student is 
currently learning to something foundational or previously covered.

Rules:
- One quick connection or summary
- Reference what's specifically on their screen
- Frame it as: "Remember when you learned X? This builds on that because..."
  or "This concept connects to Y — here's how..."
- 2-3 sentences max
- Don't over-explain, just spark the connection"""

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
        logger.error(f"[tool_review] Error: {e}")
        return "Review suggestion failed — try again soon."
