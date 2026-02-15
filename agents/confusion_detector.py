"""
ConfusionDetector — Pure Python algorithmic confusion scoring.
No LLM calls. Signal fusion from behavioral cues + Gemini VLM analysis.
"""
import re
from agents.models import WorkContext, ConfusionAssessment
from agents.config import (
    WEIGHT_TYPING, WEIGHT_DELETION, WEIGHT_PAUSE,
    WEIGHT_REREAD, WEIGHT_VERBAL, WEIGHT_TOUCH,
    WEIGHT_GEMINI, CONFUSION_THRESHOLD,
)

# Keywords for classification
VISUAL_KEYWORDS = {
    "visual", "visualize", "graph", "plot", "diagram", "picture",
    "image", "shape", "geometry", "spatial", "3d", "surface",
    "draw", "sketch", "see", "look like", "imagine",
}
CONCEPTUAL_KEYWORDS = {
    "why", "concept", "understand", "meaning", "intuition",
    "reason", "explain", "theory", "fundamental", "abstract",
    "idea", "principle", "logic", "proof",
}
PROCEDURAL_KEYWORDS = {
    "how", "steps", "procedure", "process", "method",
    "algorithm", "implement", "code", "solve", "calculate",
    "compute", "formula", "equation", "syntax",
}
VERBAL_CONFUSION_CUE_WEIGHTS = {
    "confused": 0.8,
    "don't understand": 0.9,
    "lost": 0.7,
    "stuck": 0.7,
    "what": 0.4,
    "huh": 0.6,
    "wait": 0.3,
    "hmm": 0.3,
    "no idea": 0.9,
    "makes no sense": 0.9,
    "help": 0.6,
}


class ConfusionDetector:
    """Algorithmic confusion detection from behavioral signals + Gemini VLM."""

    def score(self, ctx: WorkContext) -> ConfusionAssessment:
        """Score confusion level from a WorkContext."""

        # ─── Signal Extraction ───
        s_typing = self._typing_signal(ctx.typing_speed_ratio)
        s_deletion = self._deletion_signal(ctx.deletion_rate)
        s_pause = self._pause_signal(ctx.pause_duration)
        s_reread = self._reread_signal(ctx.scroll_back_count)

        # Verbal signal: check both explicit cues and user_message
        verbal_cues = list(ctx.verbal_confusion_cues or [])
        if ctx.user_message:
            verbal_cues.append(ctx.user_message)
        s_verbal = self._verbal_signal(verbal_cues)

        s_touch = 1.0 if ctx.user_touched_agent else 0.0

        # ─── Gemini VLM Signal (new!) ───
        s_gemini = self._gemini_signal(ctx)

        # ─── Weighted Fusion (dynamically normalized) ───
        # When Gemini data is present, it gets its weight. When absent,
        # behavioral signals are re-normalized to use the full range.
        has_gemini = self._has_gemini_data(ctx)

        if has_gemini:
            # Full fusion: behavioral + Gemini VLM
            confusion_score = (
                s_typing * WEIGHT_TYPING
                + s_deletion * WEIGHT_DELETION
                + s_pause * WEIGHT_PAUSE
                + s_reread * WEIGHT_REREAD
                + s_verbal * WEIGHT_VERBAL
                + s_touch * WEIGHT_TOUCH
                + s_gemini * WEIGHT_GEMINI
            )
        else:
            # Behavioral only — normalize weights to sum to 1.0
            behavioral_total = (
                WEIGHT_TYPING + WEIGHT_DELETION + WEIGHT_PAUSE
                + WEIGHT_REREAD + WEIGHT_VERBAL + WEIGHT_TOUCH
            )
            confusion_score = (
                s_typing * (WEIGHT_TYPING / behavioral_total)
                + s_deletion * (WEIGHT_DELETION / behavioral_total)
                + s_pause * (WEIGHT_PAUSE / behavioral_total)
                + s_reread * (WEIGHT_REREAD / behavioral_total)
                + s_verbal * (WEIGHT_VERBAL / behavioral_total)
                + s_touch * (WEIGHT_TOUCH / behavioral_total)
            )

        # Clamp
        confusion_score = max(0.0, min(1.0, confusion_score))

        # ─── Classification ───
        confusion_type = self._classify(
            ctx, s_typing, s_deletion, s_pause, s_reread, s_verbal, s_touch,
            s_gemini, confusion_score,
        )

        # ─── Intervention Decision ───
        # Override: if Gemini VLM is confident user is stuck or has errors, intervene
        gemini_override = bool(
            (ctx.gemini_stuck and ctx.gemini_work_status in ("incorrect", "incomplete"))
            or (ctx.gemini_work_status == "incorrect" and ctx.gemini_error)
            or (len(ctx.gemini_confused_about) >= 2)
        )

        should_intervene = (
            ctx.user_touched_agent
            or confusion_score > CONFUSION_THRESHOLD
            or gemini_override
        )

        # ─── Build Reasoning ───
        signals = {
            "typing": round(s_typing, 3),
            "deletion": round(s_deletion, 3),
            "pause": round(s_pause, 3),
            "reread": round(s_reread, 3),
            "verbal": round(s_verbal, 3),
            "touch": round(s_touch, 3),
            "gemini": round(s_gemini, 3),
        }
        top_signals = sorted(signals.items(), key=lambda x: x[1], reverse=True)[:3]
        reasoning_parts = []
        for name, val in top_signals:
            if val > 0.3:
                reasoning_parts.append(f"{name}={val}")
        reasoning = (
            f"score={confusion_score:.2f}, type={confusion_type}, "
            f"top_signals=[{', '.join(reasoning_parts)}]"
        )

        return ConfusionAssessment(
            confusion_score=round(confusion_score, 3),
            confusion_type=confusion_type,
            should_intervene=should_intervene,
            signals=signals,
            reasoning=reasoning,
        )

    # ─── Signal Functions ───

    @staticmethod
    def _typing_signal(typing_speed_ratio: float) -> float:
        """Slow typing → higher confusion signal."""
        if typing_speed_ratio <= 0:
            return 0.8
        return max(0.0, 1.0 - typing_speed_ratio)

    @staticmethod
    def _deletion_signal(deletion_rate: float) -> float:
        """High deletion rate → confusion.
        deletion_rate is a ratio 0.0–1.0 (fraction of keystrokes that were deletions).
        > 0.3 = some confusion, > 0.5 = significant, > 0.7 = very high."""
        if deletion_rate > 0.6:
            return 0.9
        elif deletion_rate > 0.4:
            return 0.6
        elif deletion_rate > 0.2:
            return 0.3
        return max(0.0, deletion_rate)

    @staticmethod
    def _pause_signal(pause_duration: float) -> float:
        """Long pauses → confusion (seconds since last keystroke)."""
        if pause_duration > 30:
            return 0.95
        elif pause_duration > 15:
            return 0.7
        elif pause_duration > 8:
            return 0.5
        elif pause_duration > 3:
            return 0.2
        return 0.0

    @staticmethod
    def _reread_signal(scroll_back_count: int) -> float:
        """Scrolling back / re-reading → confusion."""
        if scroll_back_count >= 8:
            return 1.0
        elif scroll_back_count >= 4:
            return 0.7
        elif scroll_back_count >= 2:
            return 0.4
        elif scroll_back_count >= 1:
            return 0.2
        return 0.0

    @staticmethod
    def _verbal_signal(verbal_confusion_cues: list[str]) -> float:
        """Verbal cues of confusion (from audio transcript)."""
        if not verbal_confusion_cues:
            return 0.0
        max_weight = 0.0
        for cue in verbal_confusion_cues:
            cue_lower = cue.lower().strip()
            for keyword, weight in VERBAL_CONFUSION_CUE_WEIGHTS.items():
                if keyword in cue_lower:
                    max_weight = max(max_weight, weight)
        return max_weight

    @staticmethod
    def _gemini_signal(ctx: WorkContext) -> float:
        """
        Gemini VLM signal — combines stuck detection, work correctness,
        and identified confusion topics into a single confusion score.

        This is the strongest individual signal because Gemini actually
        SEES what the user is doing, not just their keystrokes.
        """
        score = 0.0

        # Gemini says user is stuck → strong signal
        if ctx.gemini_stuck:
            score += 0.6

        # Work status: incorrect work → high confusion, incomplete → moderate
        status = ctx.gemini_work_status.lower() if ctx.gemini_work_status else ""
        if status == "incorrect":
            score += 0.7
        elif status == "incomplete":
            score += 0.2

        # Gemini identified specific confusion topics
        if ctx.gemini_confused_about:
            score += min(0.5, len(ctx.gemini_confused_about) * 0.25)

        # Gemini identified an error in their work
        if ctx.gemini_error:
            score += 0.3

        # Clamp to [0, 1]
        return min(1.0, score)

    def _classify(
        self,
        ctx: WorkContext,
        s_typing: float,
        s_deletion: float,
        s_pause: float,
        s_reread: float,
        s_verbal: float,
        s_touch: float,
        s_gemini: float,
        score: float,
    ) -> str:
        """Classify confusion type based on priority rules."""
        msg_lower = (ctx.user_message or "").lower()

        # Priority 1: Explicit touch with visual keywords
        if ctx.user_touched_agent and msg_lower and self._has_keywords(msg_lower, VISUAL_KEYWORDS):
            return "VISUAL_SPATIAL"

        # Priority 2: Explicit touch with conceptual keywords
        if ctx.user_touched_agent and msg_lower and self._has_keywords(msg_lower, CONCEPTUAL_KEYWORDS):
            return "CONCEPTUAL_WHY"

        # Priority 3: Explicit touch with procedural keywords
        if ctx.user_touched_agent and msg_lower and self._has_keywords(msg_lower, PROCEDURAL_KEYWORDS):
            return "PROCEDURAL_HOW"

        # Priority 4: Explicit touch with no message
        if ctx.user_touched_agent:
            return "CONCEPTUAL_WHY"  # default for explicit request

        # Priority 5: Gemini says work is incorrect → procedural if coding, conceptual if reading
        if ctx.gemini_work_status == "incorrect":
            if ctx.screen_content_type in ("code", "equation"):
                return "PROCEDURAL_HOW"
            else:
                return "CONCEPTUAL_WHY"

        # Priority 6: Gemini stuck + visual content → visual
        if ctx.gemini_stuck and ctx.screen_content_type in ("equation", "diagram"):
            return "VISUAL_SPATIAL"

        # Priority 7: Gemini stuck + confused about concepts → conceptual
        if ctx.gemini_stuck and ctx.gemini_confused_about:
            return "CONCEPTUAL_WHY"

        # Priority 8: Visual content + re-reading
        if ctx.screen_content_type in ("equation", "diagram") and s_reread > 0.5:
            return "VISUAL_SPATIAL"

        # Priority 9: Long pause + verbal cues
        if s_pause > 0.5 and s_verbal > 0.3:
            return "CONCEPTUAL_WHY"

        # Priority 10: High deletion rate
        if s_deletion > 0.6:
            return "PROCEDURAL_HOW"

        # Priority 11: Low score → extending (user doing well)
        if score < 0.3:
            return "NONE_EXTENDING"

        # Default
        return "CONCEPTUAL_WHY"

    @staticmethod
    def _has_gemini_data(ctx: WorkContext) -> bool:
        """Check if this WorkContext contains Gemini VLM analysis data."""
        return (
            ctx.gemini_stuck
            or ctx.gemini_work_status not in ("unclear", "")
            or len(ctx.gemini_confused_about) > 0
            or len(ctx.gemini_understands) > 0
            or bool(ctx.gemini_error)
        )

    @staticmethod
    def _has_keywords(text: str, keywords: set[str]) -> bool:
        """Check if text contains any of the keywords."""
        for kw in keywords:
            if kw in text:
                return True
        return False
