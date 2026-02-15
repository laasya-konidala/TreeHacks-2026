"""
Payment Protocol for monetization via FET tokens.

Implements the official Fetch.ai payment_protocol_spec (seller role) so
ASI:One can trigger payment flows.  Also keeps the custom tier logic
(free / premium / per_mastery) for internal feature gating.

Tiers:
  - free:        3 interventions/day, behavioral detection only
  - premium:     unlimited interventions + screen analysis + multi-turn dialogue
  - per_mastery: charge 0.1 FET when a concept's mastery crosses 0.85
"""
import os
import time
import logging
from datetime import datetime, timezone
from uuid import uuid4

from uagents import Context, Model, Protocol
from uagents_core.contrib.protocols.chat import (
    ChatMessage,
    TextContent,
    EndSessionContent,
)
from uagents_core.contrib.protocols.payment import (
    Funds,
    RequestPayment,
    RejectPayment,
    CommitPayment,
    CancelPayment,
    CompletePayment,
    payment_protocol_spec,
)

logger = logging.getLogger(__name__)

# ─── Official Payment Protocol (seller role) ───
payment_proto = Protocol(spec=payment_protocol_spec, role="seller")

# ─── Payment config ───
FET_FUNDS = Funds(currency="FET", amount="0.1", payment_method="fet_direct")
ACCEPTED_FUNDS = [FET_FUNDS]

# ─── Wallet (set from main agent file) ───
_agent_wallet = None


def set_agent_wallet(wallet):
    """Call from agent.py to inject the wallet for on-chain verification."""
    global _agent_wallet
    _agent_wallet = wallet


# ─── Helpers ───

def _create_text_chat(text: str, end_session: bool = False) -> ChatMessage:
    """Create a ChatMessage wrapping plain text."""
    content = [TextContent(type="text", text=text)]
    if end_session:
        content.append(EndSessionContent(type="end-session"))
    return ChatMessage(
        timestamp=datetime.now(timezone.utc),
        msg_id=uuid4(),
        content=content,
    )


async def request_payment_from_user(
    ctx: Context, user_address: str, description: str | None = None
):
    """Send a payment request to a user (e.g. for premium upgrade)."""
    metadata = {}
    fet_network = (
        "stable-testnet"
        if os.getenv("FET_USE_TESTNET", "true").lower() == "true"
        else "mainnet"
    )
    if _agent_wallet:
        metadata["provider_agent_wallet"] = str(_agent_wallet.address())
    metadata["fet_network"] = fet_network
    metadata["content"] = description or (
        "Upgrade to premium for unlimited interventions, screen analysis, "
        "and multi-turn dialogue. Pay 0.1 FET."
    )

    payment_request = RequestPayment(
        accepted_funds=ACCEPTED_FUNDS,
        recipient=str(_agent_wallet.address()) if _agent_wallet else "unknown",
        deadline_seconds=300,
        reference=str(uuid4()),
        description=description or "Ambient Learning — Premium upgrade (0.1 FET)",
        metadata=metadata,
    )
    ctx.logger.info(f"Sending payment request to {user_address}")
    await ctx.send(user_address, payment_request)


def verify_fet_payment(
    transaction_id: str,
    expected_amount_fet: str,
    sender_fet_address: str,
    recipient_wallet,
    ctx_logger,
) -> bool:
    """Verify an on-chain FET payment."""
    try:
        from cosmpy.aerial.client import LedgerClient, NetworkConfig

        testnet = os.getenv("FET_USE_TESTNET", "true").lower() == "true"
        network_config = (
            NetworkConfig.fetchai_stable_testnet()
            if testnet
            else NetworkConfig.fetchai_mainnet()
        )
        ledger = LedgerClient(network_config)
        expected_amount_micro = int(float(expected_amount_fet) * 10**18)

        ctx_logger.info(
            f"Verifying {expected_amount_fet} FET: "
            f"{sender_fet_address} → {recipient_wallet.address()}"
        )

        tx_response = ledger.query_tx(transaction_id)
        if not tx_response.is_successful():
            ctx_logger.error(f"Transaction {transaction_id} failed on-chain")
            return False

        denom = "atestfet" if testnet else "afet"
        expected_recipient = str(recipient_wallet.address())
        recipient_ok = amount_ok = sender_ok = False

        for event_type, event_attrs in tx_response.events.items():
            if event_type == "transfer":
                if event_attrs.get("recipient") == expected_recipient:
                    recipient_ok = True
                    if event_attrs.get("sender") == sender_fet_address:
                        sender_ok = True
                    amt_str = event_attrs.get("amount", "")
                    if amt_str and amt_str.endswith(denom):
                        try:
                            if int(amt_str.replace(denom, "")) >= expected_amount_micro:
                                amount_ok = True
                        except Exception:
                            pass

        if recipient_ok and amount_ok and sender_ok:
            ctx_logger.info(f"Payment verified: {transaction_id}")
            return True

        ctx_logger.error(
            f"Verification failed — recipient:{recipient_ok} "
            f"amount:{amount_ok} sender:{sender_ok}"
        )
        return False
    except Exception as e:
        ctx_logger.error(f"FET payment verification error: {e}")
        return False


# ─── Official Payment Protocol Handlers ───

@payment_proto.on_message(CommitPayment)
async def handle_commit_payment(ctx: Context, sender: str, msg: CommitPayment):
    """Buyer committed payment — verify on-chain and complete."""
    ctx.logger.info(f"Payment commitment from {sender}")
    verified = False

    if msg.funds.payment_method == "fet_direct" and msg.funds.currency == "FET":
        buyer_wallet = None
        if isinstance(msg.metadata, dict):
            buyer_wallet = (
                msg.metadata.get("buyer_fet_wallet")
                or msg.metadata.get("buyer_fet_address")
            )
        if not buyer_wallet:
            ctx.logger.error("Missing buyer_fet_wallet in CommitPayment metadata")
        elif _agent_wallet:
            verified = verify_fet_payment(
                transaction_id=msg.transaction_id,
                expected_amount_fet=FET_FUNDS.amount,
                sender_fet_address=buyer_wallet,
                recipient_wallet=_agent_wallet,
                ctx_logger=ctx.logger,
            )
    else:
        ctx.logger.error(f"Unsupported payment: {msg.funds.payment_method}")

    if verified:
        ctx.logger.info(f"Payment verified from {sender} — upgrading to premium")
        await ctx.send(sender, CompletePayment(transaction_id=msg.transaction_id))

        # Upgrade user tier
        user_state = _get_user_state(sender)
        user_state["tier"] = "premium"

        await ctx.send(
            sender,
            _create_text_chat(
                "Payment received! You've been upgraded to Premium. "
                "You now have unlimited interventions, screen analysis, "
                "and multi-turn dialogue. Happy learning!"
            ),
        )
    else:
        ctx.logger.error(f"Payment verification failed from {sender}")
        await ctx.send(
            sender,
            CancelPayment(
                transaction_id=msg.transaction_id,
                reason="Payment verification failed. Please try again.",
            ),
        )


@payment_proto.on_message(RejectPayment)
async def handle_reject_payment(ctx: Context, sender: str, msg: RejectPayment):
    """Buyer rejected the payment request."""
    ctx.logger.info(f"Payment rejected by {sender}: {msg.reason}")
    await ctx.send(
        sender,
        _create_text_chat(
            "No worries! You can continue using the free tier "
            "(3 interventions per day). Send another message anytime "
            "to request an upgrade."
        ),
    )


# ─── Custom Tier Management (kept for internal feature gating) ───

# In-memory state (hackathon)
user_tiers: dict[str, dict] = {}

TIER_LIMITS = {
    "free": {"interventions_per_day": 3, "screen_analysis": False, "multi_turn": False},
    "premium": {"interventions_per_day": 999, "screen_analysis": True, "multi_turn": True},
    "per_mastery": {"interventions_per_day": 999, "screen_analysis": True, "multi_turn": True},
}


def _get_user_state(user_id: str) -> dict:
    """Get or create user payment state."""
    if user_id not in user_tiers:
        user_tiers[user_id] = {
            "tier": "free",
            "interventions_today": 0,
            "last_reset": time.strftime("%Y-%m-%d"),
            "total_mastered": 0,
            "asi_balance": 0.0,
        }
    today = time.strftime("%Y-%m-%d")
    if user_tiers[user_id]["last_reset"] != today:
        user_tiers[user_id]["interventions_today"] = 0
        user_tiers[user_id]["last_reset"] = today
    return user_tiers[user_id]


def check_can_intervene(user_id: str) -> bool:
    """Check if a user can receive an intervention."""
    state = _get_user_state(user_id)
    limit = TIER_LIMITS.get(state["tier"], TIER_LIMITS["free"])["interventions_per_day"]
    return state["interventions_today"] < limit


def record_intervention(user_id: str) -> None:
    """Record that an intervention was used."""
    state = _get_user_state(user_id)
    state["interventions_today"] += 1


def can_use_screen_analysis(user_id: str) -> bool:
    """Check if user's tier allows screen analysis."""
    state = _get_user_state(user_id)
    return TIER_LIMITS.get(state["tier"], TIER_LIMITS["free"])["screen_analysis"]


def can_use_multi_turn(user_id: str) -> bool:
    """Check if user's tier allows multi-turn dialogue."""
    state = _get_user_state(user_id)
    return TIER_LIMITS.get(state["tier"], TIER_LIMITS["free"])["multi_turn"]


# ─── Custom Protocol for tier queries (internal agent-to-agent) ───

class SetPaymentTier(Model):
    """Set the user's payment tier."""
    tier: str  # "free" | "premium" | "per_mastery"
    user_id: str = "default"


class GetPaymentStatus(Model):
    """Query current payment status."""
    user_id: str = "default"


class PaymentStatus(Model):
    """Current payment status response."""
    active: bool
    tier: str
    interventions_today: int
    interventions_limit: int
    total_concepts_mastered: int
    asi_balance: float


# Custom protocol for internal tier management
tier_protocol = Protocol(name="learning_payment_tiers", version="0.1.0")


@tier_protocol.on_message(model=SetPaymentTier, replies={PaymentStatus})
async def handle_set_tier(ctx: Context, sender: str, msg: SetPaymentTier):
    """Set a user's payment tier."""
    user_id = msg.user_id or sender
    state = _get_user_state(user_id)

    if msg.tier not in TIER_LIMITS:
        ctx.logger.warning(f"Unknown tier '{msg.tier}' from {sender}")
        msg.tier = "free"

    state["tier"] = msg.tier
    ctx.logger.info(f"Payment tier set: user={user_id}, tier={msg.tier}")

    limit = TIER_LIMITS[state["tier"]]["interventions_per_day"]
    await ctx.send(sender, PaymentStatus(
        active=True,
        tier=state["tier"],
        interventions_today=state["interventions_today"],
        interventions_limit=limit,
        total_concepts_mastered=state["total_mastered"],
        asi_balance=state["asi_balance"],
    ))


@tier_protocol.on_message(model=GetPaymentStatus, replies={PaymentStatus})
async def handle_get_status(ctx: Context, sender: str, msg: GetPaymentStatus):
    """Query a user's payment status."""
    user_id = msg.user_id or sender
    state = _get_user_state(user_id)
    limit = TIER_LIMITS[state["tier"]]["interventions_per_day"]

    await ctx.send(sender, PaymentStatus(
        active=True,
        tier=state["tier"],
        interventions_today=state["interventions_today"],
        interventions_limit=limit,
        total_concepts_mastered=state["total_mastered"],
        asi_balance=state["asi_balance"],
    ))
