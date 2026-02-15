"""
Tool: Quiz / Comprehension Questions

Generates contextual questions calibrated to mastery level.
Used by all three agents but with different framing:
  - Conceptual: "do you understand what you're seeing?"
  - Applied: "can you solve this?" (TODO)
  - Extension: "what if we changed this?" (TODO)
"""
import logging
import anthropic

from agents.config import ANTHROPIC_API_KEY, CLAUDE_MODEL

logger = logging.getLogger(__name__)

_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


async def generate_question(
    vlm_context: str,
    topic: str,
    mastery_pct: int,
    speech_context: str = "",
    agent_framing: str = "conceptual",
) -> str:
    """Generate a single contextual question based on what's on screen."""
    framing = {
        "conceptual": (
            "You're a Socratic study buddy. The student is watching/reading and building understanding. "
            "Ask ONE question that checks if they truly grasp what's on their screen."
        ),
        "applied": (
            "You're a practice coach. The student is working on a problem. "
            "Ask ONE question that nudges them toward the right approach without giving the answer."
        ),
        "extension": (
            "You're an intellectual challenger. Push the student beyond the basics. "
            "Ask ONE 'what if' or 'why not' question that deepens their understanding."
        ),
    }

    difficulty = {
        range(0, 31): "basic — ask about definitions or 'what is' concepts",
        range(31, 71): "intermediate — ask 'why' or 'how does this relate to' questions",
        range(71, 101): "advanced — ask 'what would happen if' or edge-case questions",
    }

    diff_desc = "intermediate"
    for r, desc in difficulty.items():
        if mastery_pct in r:
            diff_desc = desc
            break

    system = framing.get(agent_framing, framing['conceptual']) + """

Rules:
- Reference EXACTLY what's on their screen (specific equations, diagrams, code, etc.)
- Ask ONE clear question, not multiple
- Don't give the answer
- Be concise and conversational
- 2-3 sentences max"""

    user_msg = f"""What's on the student's screen right now:
{vlm_context}

Their mastery of "{topic}" is {mastery_pct}%.
Difficulty level: {diff_desc}

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
        logger.error(f"[tool_quiz] Error: {e}")
        return "Keep going — I'll check in again soon!"


async def generate_quiz(
    vlm_context: str,
    topic: str,
    mastery_pct: int,
    num_questions: int = 3,
) -> str:
    """Generate a short multi-question quiz."""
    system = "You are a quiz generator for students. Create concise, contextual quizzes."
    user_msg = f"""Create a {num_questions}-question quick quiz about "{topic}" based on what's on screen.

Screen context:
{vlm_context}

Mastery: {mastery_pct}% — calibrate difficulty accordingly.

Format each question as:
Q1: [question]
A) [option]  B) [option]  C) [option]  D) [option]

Keep questions contextual to what's on screen. Be concise."""

    try:
        response = _client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=500,
            system=system,
            messages=[{"role": "user", "content": user_msg}],
        )
        return response.content[0].text
    except Exception as e:
        logger.error(f"[tool_quiz] Quiz error: {e}")
        return "Quiz generation failed — try again in a moment."
