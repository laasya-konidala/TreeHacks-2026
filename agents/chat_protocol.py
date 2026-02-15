"""
ASI:One-compatible Chat Protocol for the Ambient Learning Agent System.

Uses the standard uagents_core chat_protocol_spec so that ASI:One can
discover and route queries to our orchestrator.  Every incoming ChatMessage
is treated as a tutoring help request — we generate a response via Gemini
and send it back as a ChatMessage.
"""
import logging
from datetime import datetime, timezone
from uuid import uuid4

from uagents import Context, Protocol
from uagents_core.contrib.protocols.chat import (
    ChatAcknowledgement,
    ChatMessage,
    TextContent,
    StartSessionContent,
    EndSessionContent,
    chat_protocol_spec,
)

logger = logging.getLogger(__name__)

# ─── Protocol (ASI:One compatible) ───
chat_proto = Protocol(spec=chat_protocol_spec)


# ─── Helpers ───

def create_text_chat(text: str, end_session: bool = False) -> ChatMessage:
    """Create a ChatMessage wrapping plain text."""
    content = [TextContent(type="text", text=text)]
    if end_session:
        content.append(EndSessionContent(type="end-session"))
    return ChatMessage(
        timestamp=datetime.now(timezone.utc),
        msg_id=uuid4(),
        content=content,
    )


SYSTEM_PROMPT = (
    "You are an ambient learning tutor. A user found you on ASI:One "
    "and is asking for help with studying. Be concise, friendly, and "
    "helpful. If they ask about a concept, explain it clearly with an "
    "analogy. If they ask how to do something, give clear steps. "
    "Keep responses under 3 paragraphs.\n\n"
    "You are part of the Ambient Learning Agent System — a multi-agent "
    "tutoring system that observes what students are working on and "
    "helps them build understanding through contextual questions, "
    "visualizations, and guided problem-solving."
)


def _generate_direct_response(user_text: str) -> str:
    """Generate a response for ASI:One chat users. Tries Claude first, falls back to Gemini."""
    # Try Claude first (reliable, high rate limits)
    try:
        import anthropic
        from agents.config import ANTHROPIC_API_KEY, CLAUDE_MODEL

        if ANTHROPIC_API_KEY:
            client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
            response = client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=500,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_text}],
            )
            return response.content[0].text
    except Exception as e:
        logger.warning(f"Claude failed for ASI:One chat, trying Gemini: {e}")

    # Fall back to Gemini
    try:
        from google import genai
        from google.genai import types
        from agents.config import GEMINI_MODEL, GEMINI_API_KEY

        client = genai.Client(api_key=GEMINI_API_KEY)
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=user_text,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                max_output_tokens=500,
            ),
        )
        return response.text or "I'm here to help you learn! Ask me anything."
    except Exception as e:
        logger.error(f"Both Claude and Gemini failed for ASI:One chat: {e}")
        return (
            "I'd love to help, but I'm having trouble connecting to my AI backend. "
            "Try again in a moment!"
        )


# ─── Message Handlers ───

@chat_proto.on_message(ChatMessage)
async def handle_message(ctx: Context, sender: str, msg: ChatMessage):
    """Handle incoming chat from ASI:One users or other agents."""
    ctx.logger.info(f"Chat message from {sender}")

    # Always acknowledge receipt immediately
    await ctx.send(
        sender,
        ChatAcknowledgement(
            timestamp=datetime.now(timezone.utc),
            acknowledged_msg_id=msg.msg_id,
        ),
    )

    # Process each content item
    for item in msg.content:
        if isinstance(item, StartSessionContent):
            ctx.logger.info(f"Session started with {sender}")
            continue

        elif isinstance(item, TextContent):
            ctx.logger.info(f"Text from {sender}: {item.text[:120]}")

            # Generate a tutoring response
            response_text = _generate_direct_response(item.text)

            # Send response back
            response = create_text_chat(response_text)
            await ctx.send(sender, response)

        elif isinstance(item, EndSessionContent):
            ctx.logger.info(f"Session ended with {sender}")

        else:
            ctx.logger.info(f"Unexpected content type from {sender}")


@chat_proto.on_message(ChatAcknowledgement)
async def handle_ack(ctx: Context, sender: str, msg: ChatAcknowledgement):
    """Handle acknowledgement for messages we sent."""
    ctx.logger.info(
        f"Acknowledged by {sender} for msg_id={msg.acknowledged_msg_id}"
    )
