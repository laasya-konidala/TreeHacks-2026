"""
Unit tests for ConfidenceWeightedBKT.
Tests the key innovation: confidence-weighted observations.
"""
import sys
import os
import pytest

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agents.learner_model import ConfidenceWeightedBKT


class TestBKTBasics:
    """Basic BKT behavior tests."""

    def test_init_concept(self):
        bkt = ConfidenceWeightedBKT()
        bkt.init_concept("linear_algebra")
        assert bkt.get_mastery("linear_algebra") == 0.3  # default prior

    def test_unknown_concept_returns_zero(self):
        bkt = ConfidenceWeightedBKT()
        assert bkt.get_mastery("nonexistent") == 0.0

    def test_correct_increases_mastery(self):
        bkt = ConfidenceWeightedBKT()
        bkt.init_concept("calc")
        before = bkt.get_mastery("calc")
        bkt.update("calc", correct=True, confidence=1.0)
        after = bkt.get_mastery("calc")
        assert after > before, f"Expected mastery to increase: {before} -> {after}"

    def test_incorrect_decreases_mastery(self):
        bkt = ConfidenceWeightedBKT()
        bkt.init_concept("calc")
        # First build up some mastery
        for _ in range(5):
            bkt.update("calc", correct=True, confidence=1.0)
        before = bkt.get_mastery("calc")
        bkt.update("calc", correct=False, confidence=1.0)
        after = bkt.get_mastery("calc")
        assert after < before, f"Expected mastery to decrease: {before} -> {after}"

    def test_multiple_correct_reaches_mastery(self):
        bkt = ConfidenceWeightedBKT()
        bkt.init_concept("calc")
        for _ in range(20):
            bkt.update("calc", correct=True, confidence=1.0)
        assert bkt.is_mastered("calc"), f"Expected mastered, got {bkt.get_mastery('calc')}"


class TestConfidenceWeighting:
    """Tests for the confidence weighting innovation."""

    def test_low_confidence_moves_less(self):
        bkt_high = ConfidenceWeightedBKT()
        bkt_low = ConfidenceWeightedBKT()

        bkt_high.init_concept("c1")
        bkt_low.init_concept("c1")

        bkt_high.update("c1", correct=True, confidence=1.0)
        bkt_low.update("c1", correct=True, confidence=0.3)

        high_mastery = bkt_high.get_mastery("c1")
        low_mastery = bkt_low.get_mastery("c1")

        assert high_mastery > low_mastery, (
            f"High confidence ({high_mastery}) should move mastery more "
            f"than low confidence ({low_mastery})"
        )

    def test_zero_confidence_no_change(self):
        bkt = ConfidenceWeightedBKT()
        bkt.init_concept("c1")
        before = bkt.get_mastery("c1")

        # With confidence=0, the observation should barely move mastery
        # (only the learning transition applies)
        bkt.update("c1", correct=True, confidence=0.0)
        after = bkt.get_mastery("c1")

        # The change should be very small (just p_learn effect)
        change = abs(after - before)
        # With confidence=0: weighted = p_know + 0*(posterior - p_know) = p_know
        # Then new = p_know + (1-p_know)*p_learn = 0.3 + 0.7*0.1 = 0.37
        # So change ≈ 0.07 (just learning transition, no Bayesian update)
        assert change < 0.1, f"Zero confidence should produce minimal change: {change}"

    def test_confidence_ordering(self):
        """Screen (0.7) > dialogue (0.5) > behavioral (0.35)."""
        results = {}
        for source, conf in [("screen", 0.7), ("dialogue", 0.5), ("behavioral", 0.35)]:
            bkt = ConfidenceWeightedBKT()
            bkt.init_concept("c1")
            bkt.update("c1", correct=True, confidence=conf, source=source)
            results[source] = bkt.get_mastery("c1")

        assert results["screen"] > results["dialogue"] > results["behavioral"], (
            f"Expected screen > dialogue > behavioral: {results}"
        )

    def test_negative_confidence_ordering(self):
        """High confidence incorrect should decrease more."""
        bkt_high = ConfidenceWeightedBKT()
        bkt_low = ConfidenceWeightedBKT()

        # Build up mastery first
        for bkt in [bkt_high, bkt_low]:
            bkt.init_concept("c1")
            for _ in range(5):
                bkt.update("c1", correct=True, confidence=1.0)

        before = bkt_high.get_mastery("c1")
        bkt_high.update("c1", correct=False, confidence=1.0)
        bkt_low.update("c1", correct=False, confidence=0.3)

        high_drop = before - bkt_high.get_mastery("c1")
        low_drop = before - bkt_low.get_mastery("c1")

        assert high_drop > low_drop, (
            f"High confidence incorrect should drop more: {high_drop} vs {low_drop}"
        )


class TestAdaptiveLearningRate:
    """Tests for adaptive p_learn adjustment."""

    def test_consecutive_correct_increases_p_learn(self):
        bkt = ConfidenceWeightedBKT()
        bkt.init_concept("c1")
        initial_p_learn = bkt.concepts["c1"]["p_learn"]

        # 3 consecutive correct with high confidence
        for _ in range(3):
            bkt.update("c1", correct=True, confidence=0.8)

        assert bkt.concepts["c1"]["p_learn"] > initial_p_learn, (
            f"p_learn should increase after 3 consecutive correct"
        )

    def test_consecutive_incorrect_decreases_p_learn(self):
        bkt = ConfidenceWeightedBKT()
        bkt.init_concept("c1")
        initial_p_learn = bkt.concepts["c1"]["p_learn"]

        # 3 consecutive incorrect
        for _ in range(3):
            bkt.update("c1", correct=False, confidence=0.8)

        assert bkt.concepts["c1"]["p_learn"] < initial_p_learn, (
            f"p_learn should decrease after 3 consecutive incorrect"
        )

    def test_p_learn_capped_at_04(self):
        bkt = ConfidenceWeightedBKT()
        bkt.init_concept("c1")

        # Many consecutive correct
        for _ in range(50):
            bkt.update("c1", correct=True, confidence=0.9)

        assert bkt.concepts["c1"]["p_learn"] <= 0.4, (
            f"p_learn should be capped at 0.4: {bkt.concepts['c1']['p_learn']}"
        )

    def test_p_learn_floored_at_005(self):
        bkt = ConfidenceWeightedBKT()
        bkt.init_concept("c1")

        # Many consecutive incorrect
        for _ in range(50):
            bkt.update("c1", correct=False, confidence=0.8)

        assert bkt.concepts["c1"]["p_learn"] >= 0.05, (
            f"p_learn should be floored at 0.05: {bkt.concepts['c1']['p_learn']}"
        )


class TestMasteryThreshold:
    """Tests for mastery detection."""

    def test_not_mastered_initially(self):
        bkt = ConfidenceWeightedBKT()
        bkt.init_concept("c1")
        assert not bkt.is_mastered("c1")

    def test_mastered_after_learning(self):
        bkt = ConfidenceWeightedBKT()
        bkt.init_concept("c1")
        for _ in range(20):
            bkt.update("c1", correct=True, confidence=1.0)
        assert bkt.is_mastered("c1")

    def test_custom_threshold(self):
        bkt = ConfidenceWeightedBKT()
        bkt.init_concept("c1")
        for _ in range(5):
            bkt.update("c1", correct=True, confidence=1.0)
        mastery = bkt.get_mastery("c1")
        assert bkt.is_mastered("c1", threshold=mastery - 0.01)
        assert not bkt.is_mastered("c1", threshold=mastery + 0.01)


class TestObservationQuality:
    """Tests for observation quality metrics."""

    def test_no_data(self):
        bkt = ConfidenceWeightedBKT()
        quality = bkt.get_observation_quality("nonexistent")
        assert quality["quality"] == "no_data"
        assert quality["total"] == 0

    def test_low_quality(self):
        bkt = ConfidenceWeightedBKT()
        bkt.init_concept("c1")
        bkt.update("c1", correct=True, confidence=0.2, source="behavioral")
        quality = bkt.get_observation_quality("c1")
        assert quality["quality"] == "low"
        assert quality["total"] == 1

    def test_source_breakdown(self):
        bkt = ConfidenceWeightedBKT()
        bkt.init_concept("c1")
        bkt.update("c1", correct=True, confidence=0.7, source="screen")
        bkt.update("c1", correct=True, confidence=0.5, source="dialogue")
        bkt.update("c1", correct=True, confidence=0.35, source="behavioral")
        quality = bkt.get_observation_quality("c1")
        assert quality["source_breakdown"]["screen"] == 1
        assert quality["source_breakdown"]["dialogue"] == 1
        assert quality["source_breakdown"]["behavioral"] == 1

    def test_high_quality(self):
        bkt = ConfidenceWeightedBKT()
        bkt.init_concept("c1")
        for _ in range(5):
            bkt.update("c1", correct=True, confidence=0.8, source="screen")
        quality = bkt.get_observation_quality("c1")
        assert quality["quality"] == "high"


class TestAutoInitialization:
    """Test that update auto-initializes unknown concepts."""

    def test_auto_init_on_update(self):
        bkt = ConfidenceWeightedBKT()
        # Should not raise, should auto-init
        bkt.update("new_concept", correct=True, confidence=0.5)
        assert bkt.get_mastery("new_concept") > 0

    def test_get_all_concepts(self):
        bkt = ConfidenceWeightedBKT()
        bkt.init_concept("a")
        bkt.init_concept("b")
        bkt.update("c", correct=True)
        all_concepts = bkt.get_all_concepts()
        assert "a" in all_concepts
        assert "b" in all_concepts
        assert "c" in all_concepts


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
