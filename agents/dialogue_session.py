"""
DialogueSession — Multi-turn state machine for Deep Diver conversations.

States: initiating → exploring → explaining → checking → closing
The state machine controls WHAT kind of response to generate.
The LLM generates words WITHIN the constraints the state machine sets.
"""
import time
from typing import Optional

from agents.config import (
    MAX_DIALOGUE_TURNS,
    MAX_DIALOGUE_DURATION_SECONDS,
    CLOSING_COMPREHENSION_STREAK,
    CLOSING_COMPREHENSION_THRESHOLD,
)


class DialogueSession:
    """Manages multi-turn dialogue state for a single tutoring session."""

    def __init__(self, session_id: str, user_id: str, trigger_context: dict):
        self.session_id = session_id
        self.user_id = user_id
        self.trigger_context = trigger_context
        self.turns: list[dict] = []                 # {role, content, timestamp, analysis}
        self.state: str = "initiating"              # state machine position

        self.confusion_model: dict = {
            "initial_hypothesis": trigger_context.get("confusion_hypothesis", ""),
            "refined_hypothesis": "",
            "confirmed_understanding": [],
            "remaining_gaps": [],
            "approaches_tried": [],
        }

        self.comprehension_signals: list[float] = []  # per-turn float 0-1
        self.started_at: float = time.time()
        self.max_turns: int = MAX_DIALOGUE_TURNS
        self.concept: str = trigger_context.get("concept", "")

    @property
    def turn_count(self) -> int:
        return len(self.turns)

    @property
    def elapsed(self) -> float:
        return time.time() - self.started_at

    def add_agent_turn(self, content: str) -> None:
        """Record an agent response turn."""
        self.turns.append({
            "role": "agent",
            "content": content,
            "timestamp": time.time(),
            "analysis": None,
        })

    def add_user_turn(self, content: str, analysis: Optional[dict] = None) -> None:
        """Record a user response turn with optional analysis."""
        self.turns.append({
            "role": "user",
            "content": content,
            "timestamp": time.time(),
            "analysis": analysis,
        })

        # Extract comprehension from analysis
        if analysis:
            comp = analysis.get("comprehension", 0.5)
            self.comprehension_signals.append(comp)

            # Update confusion model from analysis
            if analysis.get("misconception_detected"):
                self.confusion_model["remaining_gaps"].append(
                    analysis["misconception_detected"]
                )
            if analysis.get("restated_in_own_words"):
                self.confusion_model["confirmed_understanding"].append(content[:100])
            if analysis.get("remaining_confusion"):
                self.confusion_model["refined_hypothesis"] = analysis["remaining_confusion"]

    def get_next_state(self) -> str:
        """Determine the next state based on current state and signals."""
        if self.should_close():
            return "closing"

        if self.state == "initiating":
            return "exploring"

        if self.state == "exploring":
            if self.confusion_model["refined_hypothesis"]:
                return "explaining"
            return "exploring"

        if self.state == "explaining":
            if self.comprehension_signals and self.comprehension_signals[-1] > 0.6:
                return "checking"
            return "explaining"

        if self.state == "checking":
            if self.comprehension_signals and self.comprehension_signals[-1] > 0.7:
                return "closing"
            return "explaining"  # comprehension dropped, go back to explaining

        return self.state

    def advance_state(self) -> str:
        """Advance to the next state and return it."""
        self.state = self.get_next_state()
        return self.state

    def should_close(self) -> bool:
        """Check if the session should be closed."""
        # Max turns reached
        if self.turn_count >= self.max_turns:
            return True

        # Timeout
        if self.elapsed > MAX_DIALOGUE_DURATION_SECONDS:
            return True

        # Comprehension streak: 3+ consecutive above threshold
        if len(self.comprehension_signals) >= CLOSING_COMPREHENSION_STREAK:
            recent = self.comprehension_signals[-CLOSING_COMPREHENSION_STREAK:]
            if all(c > CLOSING_COMPREHENSION_THRESHOLD for c in recent):
                return True

        return False

    def get_close_reason(self) -> str:
        """Get the reason for closing."""
        if self.turn_count >= self.max_turns:
            return "max_turns"
        if self.elapsed > MAX_DIALOGUE_DURATION_SECONDS:
            return "timeout"
        if len(self.comprehension_signals) >= CLOSING_COMPREHENSION_STREAK:
            recent = self.comprehension_signals[-CLOSING_COMPREHENSION_STREAK:]
            if all(c > CLOSING_COMPREHENSION_THRESHOLD for c in recent):
                return "mastered"
        return "user_closed"

    def get_dialogue_for_prompt(self) -> str:
        """Format dialogue history for inclusion in LLM prompts."""
        lines = []
        for turn in self.turns:
            role = "Student" if turn["role"] == "user" else "Tutor"
            lines.append(f"{role}: {turn['content']}")
        return "\n".join(lines)

    def get_final_comprehension(self) -> float:
        """Get the final comprehension level."""
        if not self.comprehension_signals:
            return 0.0
        # Weighted average of last 3 signals
        recent = self.comprehension_signals[-3:]
        weights = list(range(1, len(recent) + 1))  # more recent = higher weight
        total_weight = sum(weights)
        weighted_sum = sum(s * w for s, w in zip(recent, weights))
        return round(weighted_sum / total_weight, 3)

    def get_observations(self) -> list[dict]:
        """Get all observations from this session for BKT update."""
        observations = []
        for turn in self.turns:
            if turn["role"] == "user" and turn.get("analysis"):
                analysis = turn["analysis"]
                comp = analysis.get("comprehension", 0.5)
                observations.append({
                    "concept_id": self.concept,
                    "correct": comp > 0.5,
                    "confidence": self._state_confidence(self.state),
                    "source": "dialogue",
                })
                if analysis.get("misconception_detected"):
                    observations.append({
                        "concept_id": self.concept,
                        "correct": False,
                        "confidence": 0.9,
                        "source": "dialogue_misconception",
                    })
                if analysis.get("restated_in_own_words"):
                    observations.append({
                        "concept_id": self.concept,
                        "correct": True,
                        "confidence": 0.85,
                        "source": "dialogue_restatement",
                    })
        return observations

    @staticmethod
    def _state_confidence(state: str) -> float:
        """Confidence level for observations from different states."""
        return {
            "initiating": 0.2,
            "exploring": 0.3,
            "explaining": 0.5,
            "checking": 0.85,
            "closing": 0.8,
        }.get(state, 0.3)
