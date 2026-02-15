"""
Payment Protocol for monetization via ASI tokens.

Tiers:
  - free:        3 interventions/day, behavioral detection only
  - premium:     unlimited interventions + screen analysis + multi-turn dialogue
  - per_mastery: charge 0.1 ASI when a concept's mastery crosses 0.85

This protocol lets ASI:One users query their payment status and set their tier.
"""
import time
import logging

from uagents import Context, Model, Protocol

logger = logging.getLogger(__name__)

# ─── Protocol Definition ───
payment_protocol = Protocol(name="learning_payment", version="0.1.0")

# ─── State (in-memory for hackathon) ───
user_tiers: dict[str, dict] = {}


# ─── Message Models ───

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


class PaymentReceipt(Model):
    """Receipt for a mastery-based charge."""
    concept: str
    amount: float
    mastery_level: float
    timestamp: str


# ─── Tier Configuration ───
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
    # Reset daily counter
    today = time.strftime("%Y-%m-%d")
    if user_tiers[user_id]["last_reset"] != today:
        user_tiers[user_id]["interventions_today"] = 0
        user_tiers[user_id]["last_reset"] = today
    return user_tiers[user_id]


def check_can_intervene(user_id: str) -> bool:
    """Check if a user can receive an intervention (under their tier limit)."""
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


# ─── Protocol Handlers ───

@payment_protocol.on_message(model=SetPaymentTier, replies={PaymentStatus})
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


@payment_protocol.on_message(model=GetPaymentStatus, replies={PaymentStatus})
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
