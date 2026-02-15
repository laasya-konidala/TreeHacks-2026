"""
Chat Protocol for ASI:One discovery.

Uses uagents Dialogue with the proper Edge-based handler pattern.
When ASI:One users discover this agent, they can chat with it directly.
Every incoming message is treated as an explicit tutoring help request.

Edge-based pattern:
  1. Define Nodes (states)
  2. Define Edges (transitions) with .model and .func set
  3. Construct Dialogue (auto-registers handlers from edges)
"""
import time
import logging
from datetime import datetime

from uagents import Context, Model
from uagents.experimental.dialogues import Dialogue, Edge, Node

logger = logging.getLogger(__name__)


# ─── Message Models ───

class ChatMessage(Model):
    """Incoming chat from ASI:One user."""
    text: str
    user_id: str = ""
    timestamp: str = ""


class ChatResponse(Model):
    """Agent's response back to ASI:One user."""
    text: str
    agent_type: str = "orchestrator"
    session_id: str = ""


class ChatEnd(Model):
    """Signals end of dialogue."""
    reason: str = "completed"


# ─── Handler Functions (must be defined before Edge assignment) ───

async def handle_chat_message(ctx: Context, sender: str, msg: ChatMessage):
    """Handle incoming chat from ASI:One users."""
    logger.info(f"ASI:One chat from {sender}: {msg.text[:100]}")

    from agents.confusion_detector import ConfusionDetector
    from agents.models import WorkContext

    work_ctx = WorkContext(
        screen_content=msg.text,
        screen_content_type="text",
        detected_topic="",
        detected_subtopic="",
        typing_speed_ratio=1.0,
        deletion_rate=0.0,
        pause_duration=0.0,
        scroll_back_count=0,
        user_touched_agent=True,
        user_message=msg.text,
        user_id=msg.user_id or sender,
        session_id=f"asione_{sender}_{int(time.time())}",
        timestamp=msg.timestamp or datetime.now().isoformat(),
    )

    detector = ConfusionDetector()
    assessment = detector.score(work_ctx)

    response_text = _generate_direct_response(msg.text, assessment.confusion_type)

    await ctx.send(
        sender,
        ChatResponse(
            text=response_text,
            agent_type="orchestrator",
            session_id=work_ctx.session_id,
        ),
    )


async def handle_chat_response(ctx: Context, sender: str, msg: ChatResponse):
    """Handle response acknowledgment (for dialogue state tracking)."""
    logger.info(f"Chat response sent to {sender}: {msg.text[:50]}...")


async def handle_chat_end(ctx: Context, sender: str, msg: ChatEnd):
    """Handle end of chat dialogue."""
    logger.info(f"Chat ended with {sender}: {msg.reason}")


# ─── Dialogue State Machine ───

# Nodes (states)
default_node = Node(name="default", description="Idle — waiting for user", initial=True)
chatting_node = Node(name="chatting", description="Active tutoring session")
end_node = Node(name="end", description="Session completed")

# Edges (transitions) — assign model + func BEFORE constructing Dialogue
init_edge = Edge(
    name="start_chat",
    description="User initiates a tutoring request",
    parent=None,
    child=chatting_node,
)
init_edge.model = ChatMessage
init_edge.func = handle_chat_message

continue_edge = Edge(
    name="continue_chat",
    description="User sends follow-up message",
    parent=chatting_node,
    child=chatting_node,
)
continue_edge.model = ChatMessage
continue_edge.func = handle_chat_message

respond_edge = Edge(
    name="respond",
    description="Agent sends tutoring response",
    parent=chatting_node,
    child=chatting_node,
)
respond_edge.model = ChatResponse
respond_edge.func = handle_chat_response

end_edge = Edge(
    name="end_chat",
    description="End the tutoring session",
    parent=chatting_node,
    child=end_node,
)
end_edge.model = ChatEnd
end_edge.func = handle_chat_end

# Build the dialogue (auto-registers handlers from edges)
chat_dialogue = Dialogue(
    name="ambient_learning_chat",
    version="0.1.0",
    nodes=[default_node, chatting_node, end_node],
    edges=[init_edge, continue_edge, respond_edge, end_edge],
)


# ─── Helper ───

def _generate_direct_response(user_text: str, confusion_type: str) -> str:
    """Generate a direct response for ASI:One chat users via Gemini."""
    try:
        from google import genai
        from google.genai import types
        from agents.config import GEMINI_MODEL, GEMINI_API_KEY

        client = genai.Client(api_key=GEMINI_API_KEY)
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=user_text,
            config=types.GenerateContentConfig(
                system_instruction=(
                    "You are an ambient learning tutor. A user found you on ASI:One "
                    "and is asking for help. Be concise, friendly, and helpful. "
                    "If they ask about a concept, explain it clearly with an analogy. "
                    "If they ask how to do something, give clear steps. "
                    "Keep responses under 3 paragraphs."
                ),
                max_output_tokens=300,
            ),
        )
        return response.text or ""
    except Exception as e:
        logger.error(f"Gemini API call failed for ASI:One chat: {e}")
        return (
            f"I detected this as a {confusion_type.replace('_', ' ').lower()} question. "
            "I'd love to help, but I'm having trouble connecting to my AI backend. "
            "Try again in a moment, or use the Chrome extension + Electron overlay "
            "for the full ambient learning experience!"
        )
