"""
ConfidenceWeightedBKT — Bayesian Knowledge Tracing with confidence weighting.

Key innovation: observation confidence weights how much each observation
moves the mastery estimate. Screen analysis (0.6-0.8) > dialogue (0.5-0.85)
> behavioral (0.35). Low-confidence observations barely move the needle.
"""
import time
from typing import Optional
from agents.config import (
    BKT_DEFAULT_PRIOR, BKT_P_LEARN, BKT_P_GUESS,
    BKT_P_SLIP, BKT_MASTERY_THRESHOLD,
)


class ConfidenceWeightedBKT:
    """Bayesian Knowledge Tracing with confidence-weighted observations."""

    def __init__(self):
        self.concepts: dict[str, dict] = {}

    def init_concept(self, concept_id: str, prior: float = BKT_DEFAULT_PRIOR) -> None:
        """Initialize a concept with default BKT parameters."""
        if concept_id in self.concepts:
            return
        self.concepts[concept_id] = {
            "p_know": prior,
            "p_learn": BKT_P_LEARN,
            "p_guess": BKT_P_GUESS,
            "p_slip": BKT_P_SLIP,
            "observations": [],
            "created_at": time.time(),
            "updated_at": time.time(),
        }

    def update(
        self,
        concept_id: str,
        correct: bool,
        confidence: float = 1.0,
        source: str = "unknown",
    ) -> float:
        """
        Update mastery estimate for a concept given an observation.

        Args:
            concept_id: The concept being observed
            correct: Whether the observation indicates correct understanding
            confidence: How confident we are in this observation (0.0-1.0)
            source: Where the observation came from (screen, dialogue, behavioral)

        Returns:
            Updated mastery probability
        """
        if concept_id not in self.concepts:
            self.init_concept(concept_id)

        c = self.concepts[concept_id]
        p_know = c["p_know"]
        p_guess = c["p_guess"]
        p_slip = c["p_slip"]
        p_learn = c["p_learn"]

        # Step 1: Standard BKT posterior via Bayes theorem
        if correct:
            # P(know | correct) = P(correct | know) * P(know) / P(correct)
            p_correct_given_know = 1.0 - p_slip
            p_correct_given_not_know = p_guess
            p_correct = p_correct_given_know * p_know + p_correct_given_not_know * (1.0 - p_know)
            if p_correct > 0:
                raw_posterior = (p_correct_given_know * p_know) / p_correct
            else:
                raw_posterior = p_know
        else:
            # P(know | incorrect) = P(incorrect | know) * P(know) / P(incorrect)
            p_incorrect_given_know = p_slip
            p_incorrect_given_not_know = 1.0 - p_guess
            p_incorrect = p_incorrect_given_know * p_know + p_incorrect_given_not_know * (1.0 - p_know)
            if p_incorrect > 0:
                raw_posterior = (p_incorrect_given_know * p_know) / p_incorrect
            else:
                raw_posterior = p_know

        # Step 2: Confidence interpolation
        # confidence=0 → no change, confidence=1 → full BKT update
        weighted = p_know + confidence * (raw_posterior - p_know)

        # Step 3: Learning transition
        new_p_know = weighted + (1.0 - weighted) * p_learn

        # Clamp to valid range
        new_p_know = max(0.001, min(0.999, new_p_know))

        # Store observation
        obs = {
            "correct": correct,
            "confidence": confidence,
            "source": source,
            "timestamp": time.time(),
            "p_know_before": p_know,
            "p_know_after": new_p_know,
        }
        c["observations"].append(obs)

        # Step 4: Adaptive learning rate
        recent_obs = c["observations"][-3:]
        if len(recent_obs) >= 3:
            all_correct = all(o["correct"] for o in recent_obs)
            all_incorrect = all(not o["correct"] for o in recent_obs)
            avg_confidence = sum(o["confidence"] for o in recent_obs) / len(recent_obs)

            if all_correct and avg_confidence > 0.5:
                c["p_learn"] = min(0.4, c["p_learn"] * 1.2)
            elif all_incorrect:
                c["p_learn"] = max(0.05, c["p_learn"] * 0.8)

        # Update state
        c["p_know"] = new_p_know
        c["updated_at"] = time.time()

        return new_p_know

    def get_mastery(self, concept_id: str) -> float:
        """Get current mastery probability for a concept."""
        if concept_id not in self.concepts:
            return 0.0
        return self.concepts[concept_id]["p_know"]

    def is_mastered(self, concept_id: str, threshold: float = BKT_MASTERY_THRESHOLD) -> bool:
        """Check if a concept is mastered (above threshold)."""
        return self.get_mastery(concept_id) >= threshold

    def get_observation_quality(self, concept_id: str) -> dict:
        """Get quality metrics for observations on a concept."""
        if concept_id not in self.concepts:
            return {
                "total": 0,
                "avg_confidence": 0.0,
                "source_breakdown": {},
                "quality": "no_data",
            }

        obs = self.concepts[concept_id]["observations"]
        if not obs:
            return {
                "total": 0,
                "avg_confidence": 0.0,
                "source_breakdown": {},
                "quality": "no_data",
            }

        total = len(obs)
        avg_confidence = sum(o["confidence"] for o in obs) / total

        # Source breakdown
        sources: dict[str, int] = {}
        for o in obs:
            src = o["source"]
            sources[src] = sources.get(src, 0) + 1

        # Quality rating
        if avg_confidence >= 0.7 and total >= 5:
            quality = "high"
        elif avg_confidence >= 0.4 and total >= 3:
            quality = "medium"
        elif total >= 1:
            quality = "low"
        else:
            quality = "no_data"

        return {
            "total": total,
            "avg_confidence": round(avg_confidence, 3),
            "source_breakdown": sources,
            "quality": quality,
        }

    def get_all_concepts(self) -> dict[str, float]:
        """Get mastery levels for all tracked concepts."""
        return {cid: c["p_know"] for cid, c in self.concepts.items()}
