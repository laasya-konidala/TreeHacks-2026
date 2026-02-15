"""
ObservationPipeline — Three observation sources feed into BKT.

Sources:
  1. ScreenObservation  — Claude Vision API analysis (confidence 0.6-0.8)
  2. DialogueObservation — Multi-turn dialogue signals (confidence 0.2-0.85)
  3. BehavioralObservation — Typing/deletion/pause signals (confidence 0.35)
"""
import time
import json
import logging
from typing import Optional

from agents.learner_model import ConfidenceWeightedBKT

logger = logging.getLogger(__name__)


class ScreenObservation:
    """Observation from Claude Vision screenshot analysis."""

    def __init__(self):
        self._last_analysis_time: float = 0.0
        self._min_interval: float = 10.0  # seconds

    async def analyze(
        self,
        screenshot_b64: str,
        context: dict,
    ) -> list[dict]:
        """
        Analyze a screenshot via Claude Vision API.
        Returns list of observations for BKT.
        Rate limited to once per 10 seconds.
        """
        now = time.time()
        if now - self._last_analysis_time < self._min_interval:
            return []

        self._last_analysis_time = now

        try:
            from input_pipeline.screen_analyzer import analyze_screenshot
            result = await analyze_screenshot(screenshot_b64)
        except Exception as e:
            logger.warning(f"Screen analysis failed: {e}")
            return []

        if not result or not result.get("concept_id"):
            return []

        observations = []

        # Main concept observation
        work_status = result.get("work_status", "unclear")
        if work_status in ("correct", "incorrect"):
            correct = work_status == "correct"
            raw_confidence = result.get("confidence", 0.7)
            # Map to 0.6-0.8 range for screen observations
            confidence = 0.6 + (raw_confidence * 0.2)
            observations.append({
                "concept_id": result["concept_id"],
                "correct": correct,
                "confidence": round(confidence, 2),
                "source": "screen",
                "details": result.get("specific_error"),
            })

        # Sub-concepts demonstrated
        for concept in result.get("demonstrates_understanding_of", []):
            observations.append({
                "concept_id": concept,
                "correct": True,
                "confidence": 0.65,
                "source": "screen",
            })

        for concept in result.get("demonstrates_confusion_about", []):
            observations.append({
                "concept_id": concept,
                "correct": False,
                "confidence": 0.65,
                "source": "screen",
            })

        return observations


class DialogueObservation:
    """Observation from multi-turn dialogue interaction."""

    # Confidence by dialogue state
    STATE_CONFIDENCE = {
        "initiating": 0.2,
        "exploring": 0.3,
        "explaining": 0.5,
        "checking": 0.85,
        "closing": 0.8,
    }

    @staticmethod
    def from_turn(
        user_text: str,
        analysis: dict,
        concept: str,
        dialogue_state: str,
    ) -> list[dict]:
        """
        Extract observations from a dialogue turn.

        Args:
            user_text: What the user said
            analysis: Claude's structured analysis of the user's response
            concept: The concept being discussed
            dialogue_state: Current state machine position
        """
        observations = []
        base_confidence = DialogueObservation.STATE_CONFIDENCE.get(dialogue_state, 0.3)

        # Comprehension level
        comprehension = analysis.get("comprehension", 0.5)
        correct = comprehension > 0.5
        observations.append({
            "concept_id": concept,
            "correct": correct,
            "confidence": base_confidence,
            "source": "dialogue",
            "state": dialogue_state,
        })

        # Misconception detection → strong negative signal
        misconception = analysis.get("misconception_detected")
        if misconception:
            observations.append({
                "concept_id": concept,
                "correct": False,
                "confidence": 0.9,
                "source": "dialogue_misconception",
                "details": misconception,
            })

        # Restatement in own words → strong positive signal
        if analysis.get("restated_in_own_words"):
            observations.append({
                "concept_id": concept,
                "correct": True,
                "confidence": 0.85,
                "source": "dialogue_restatement",
            })

        return observations


class BehavioralObservation:
    """Observation from typing/deletion/pause behavioral signals."""

    @staticmethod
    def from_signals(
        topic: str,
        typing_ratio: float,
        deletion_rate: float,
        pause: float,
    ) -> Optional[dict]:
        """
        Extract observation from behavioral signals.
        ONLY emit when signal is unambiguous. Returns None for ambiguous signals.
        """
        if not topic:
            return None

        # Fluent behavior → weak positive signal
        if typing_ratio > 0.9 and deletion_rate < 2.0 and pause < 5.0:
            return {
                "concept_id": topic,
                "correct": True,
                "confidence": 0.35,
                "source": "behavioral",
            }

        # Struggling behavior → weak negative signal
        if typing_ratio < 0.3 and deletion_rate > 8.0 and pause > 20.0:
            return {
                "concept_id": topic,
                "correct": False,
                "confidence": 0.35,
                "source": "behavioral",
            }

        # Ambiguous → don't pollute BKT
        return None


class ObservationPipeline:
    """Aggregates all observation sources and feeds BKT."""

    def __init__(self, bkt: ConfidenceWeightedBKT):
        self.bkt = bkt
        self.screen_observer = ScreenObservation()
        self._observation_log: list[dict] = []

    def process_observations(self, observations: list[dict]) -> None:
        """Feed a list of observations into BKT."""
        for obs in observations:
            concept_id = obs.get("concept_id")
            if not concept_id:
                continue

            # Auto-initialize unknown concepts
            self.bkt.init_concept(concept_id)

            # Update BKT
            self.bkt.update(
                concept_id=concept_id,
                correct=obs["correct"],
                confidence=obs.get("confidence", 0.5),
                source=obs.get("source", "unknown"),
            )

            # Log
            self._observation_log.append(obs)

    async def process_screen(self, screenshot_b64: str, context: dict) -> list[dict]:
        """Process a screenshot through screen observation."""
        observations = await self.screen_observer.analyze(screenshot_b64, context)
        self.process_observations(observations)
        return observations

    def process_dialogue_turn(
        self,
        user_text: str,
        analysis: dict,
        concept: str,
        dialogue_state: str,
    ) -> list[dict]:
        """Process a dialogue turn through dialogue observation."""
        observations = DialogueObservation.from_turn(
            user_text, analysis, concept, dialogue_state,
        )
        self.process_observations(observations)
        return observations

    def process_behavioral(
        self,
        topic: str,
        typing_ratio: float,
        deletion_rate: float,
        pause: float,
    ) -> Optional[dict]:
        """Process behavioral signals."""
        obs = BehavioralObservation.from_signals(
            topic, typing_ratio, deletion_rate, pause,
        )
        if obs:
            self.process_observations([obs])
        return obs

    def get_log(self) -> list[dict]:
        """Get full observation log."""
        return self._observation_log.copy()
