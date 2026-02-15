"""
Monitoring Agent — Simulates a metrics trigger and sends ChatMessage
to the learning_orchestrator using the ASI-1 AgentChatProtocol.

Trigger → ChatMessage → Acknowledge. Nothing else.
"""
import json
import logging
from datetime import datetime, timezone
from uuid import uuid4

from uagents import Agent, Context, Protocol
from uagents_core.contrib.protocols.chat import (
    ChatAcknowledgement,
    ChatMessage,
    TextContent,
    EndSessionContent,
    chat_protocol_spec,
)

from agents.config import (
    AGENTVERSE_ENABLED,
    AGENTVERSE_URL,
)

logger = logging.getLogger(__name__)

# ─── Config ───
# The address of the learning_orchestrator agent.
# Derive it from the orchestrator seed or paste it directly.
LEARNING_ORCHESTRATOR_ADDRESS = None  # filled at startup from seed

MONITOR_SEED = "ambient_learning_monitor_seed_2026"
MONITOR_PORT = 8005

# ─── Simulated trigger flag ───
metrics_triggered = True  # flip to False to skip sending

# ─── Agent Setup ───
_mon_kwargs = dict(
    name="metrics_monitor",
    port=MONITOR_PORT,
    seed=MONITOR_SEED,
)
if AGENTVERSE_ENABLED:
    _mon_kwargs["mailbox"] = True
    _mon_kwargs["publish_agent_details"] = True
else:
    _mon_kwargs["endpoint"] = [f"http://127.0.0.1:{MONITOR_PORT}/submit"]

monitor = Agent(**_mon_kwargs)

# ─── Chat Protocol (ASI-1 compatible) ───
chat_proto = Protocol(spec=chat_protocol_spec)


def _resolve_orchestrator_address():
    """Resolve the orchestrator's agent address from its seed."""
    global LEARNING_ORCHESTRATOR_ADDRESS
    from agents.config import ORCHESTRATOR_SEED
    from uagents import Agent as _Agent

    _tmp = _Agent(name="_tmp_orch", seed=ORCHESTRATOR_SEED)
    LEARNING_ORCHESTRATOR_ADDRESS = _tmp.address
    logger.info(f"Resolved orchestrator address: {LEARNING_ORCHESTRATOR_ADDRESS}")


# ─── Startup: simulate metrics detection ───
@monitor.on_event("startup")
async def on_startup(ctx: Context):
    """Resolve addresses and send trigger if metrics_triggered is True."""
    _resolve_orchestrator_address()
    ctx.logger.info(f"Monitor agent started: {monitor.address}")

    if not metrics_triggered:
        ctx.logger.info("No metrics trigger — standing by.")
        return

    if not LEARNING_ORCHESTRATOR_ADDRESS:
        ctx.logger.error("Orchestrator address not resolved — cannot send.")
        return

    # Build the payload
    payload = json.dumps({
        "event": "metrics_triggered",
        "details": "Trigger detected in monitoring system.",
    })

    # Send ChatMessage using ASI-1 format
    msg = ChatMessage(
        timestamp=datetime.now(timezone.utc),
        msg_id=uuid4(),
        content=[
            TextContent(type="text", text=payload),
            EndSessionContent(type="end-session"),
        ],
    )

    await ctx.send(LEARNING_ORCHESTRATOR_ADDRESS, msg)
    ctx.logger.info(
        f"Sent metrics_triggered ChatMessage to orchestrator "
        f"(msg_id={msg.msg_id})"
    )


# ─── Handle acknowledgement from orchestrator ───
@chat_proto.on_message(ChatAcknowledgement)
async def handle_ack(ctx: Context, sender: str, msg: ChatAcknowledgement):
    ctx.logger.info(
        f"Acknowledged by {sender} for msg_id={msg.acknowledged_msg_id}"
    )


# ─── Handle any ChatMessage reply (optional) ───
@chat_proto.on_message(ChatMessage)
async def handle_reply(ctx: Context, sender: str, msg: ChatMessage):
    for item in msg.content:
        if isinstance(item, TextContent):
            ctx.logger.info(f"Reply from {sender}: {item.text[:200]}")

    # Acknowledge the reply
    await ctx.send(
        sender,
        ChatAcknowledgement(
            acknowledged_msg_id=msg.msg_id,
            timestamp=datetime.now(timezone.utc),
        ),
    )


# ─── Attach protocol ───
monitor.include(chat_proto, publish_manifest=True)


# ─── Standalone run ───
if __name__ == "__main__":
    monitor.run()
