"""
All uAgents message types for the Ambient Learning Agent System.
"""
from uagents import Model
from typing import Optional


class WorkContext(Model):
    """Fused context from Gemini VLM + Chrome extension behavioral signals."""
    screen_content: str = ""               # extracted text / activity description
    screen_content_type: str = "text"      # "equation" | "code" | "text" | "diagram" | "mixed"
    detected_topic: str = ""               # inferred topic (from Gemini or keyword matching)
    detected_subtopic: str = ""            # inferred subtopic
    typing_speed_ratio: float = 1.0        # current / baseline (< 0.5 = hesitating)
    deletion_rate: float = 0.0             # fraction of keystrokes that were deletions
    pause_duration: float = 0.0            # seconds since last keystroke
    scroll_back_count: int = 0             # upward scrolls in last 30s
    audio_transcript: Optional[str] = None
    verbal_confusion_cues: list[str] = []
    user_touched_agent: bool = False       # explicit help request
    user_message: Optional[str] = None     # if they typed to the agent
    screenshot_b64: Optional[str] = None   # base64 PNG (optional, Gemini handles vision now)
    user_id: str = "default"
    session_id: str = ""
    timestamp: str = ""

    # ─── Gemini VLM Analysis (from Electron screen capture) ───
    gemini_stuck: bool = False             # Gemini says user appears stuck
    gemini_work_status: str = "unclear"    # "correct" | "incorrect" | "incomplete" | "unclear"
    gemini_confused_about: list[str] = []  # concepts Gemini detected confusion about
    gemini_understands: list[str] = []     # concepts Gemini says user understands
    gemini_error: Optional[str] = None     # specific error description if work is incorrect
    gemini_mode: str = ""                  # "CONCEPTUAL" | "APPLIED" | "CONSOLIDATION"
    gemini_notes: str = ""                 # additional observations from Gemini


class ConfusionAssessment(Model):
    """Output of the ConfusionDetector — algorithmic confusion scoring."""
    confusion_score: float                 # 0.0 to 1.0
    confusion_type: str                    # CONCEPTUAL_WHY | VISUAL_SPATIAL | PROCEDURAL_HOW | NONE_EXTENDING | EXPLICIT_REQUEST
    should_intervene: bool
    signals: dict                          # individual signal scores for debugging
    reasoning: str                         # human-readable explanation


class DeepDiveRequest(Model):
    """Request to the Deep Diver agent for multi-turn conceptual dialogue."""
    concept: str
    confusion_hypothesis: str
    screen_content: str
    screen_content_type: str
    user_message: Optional[str] = None
    user_id: str = "default"
    session_id: str = ""


class DeepDiveResponse(Model):
    """Response from the Deep Diver agent."""
    message: str
    dialogue_state: str
    session_id: str
    turn_number: int
    should_close: bool = False


class UserReply(Model):
    """User's reply during a multi-turn dialogue."""
    message: str
    session_id: str
    user_id: str = "default"


class AgentMessage(Model):
    """Generic agent message sent to the UI via WebSocket."""
    content: str
    content_type: str                      # "text" | "challenge" | "visualization" | "hint"
    agent_type: str                        # "deep_diver" | "assessor" | "visualizer" | "orchestrator"
    dialogue_state: Optional[str] = None
    session_id: str = ""
    turn_number: int = 0
    metadata: dict = {}


class VisualizerRequest(Model):
    """Request to the Visualizer agent."""
    concept: str
    subconcept: str
    screen_content: str
    confusion_hypothesis: str
    user_id: str = "default"
    session_id: str = ""


class VisualizerResponse(Model):
    """Structured scene description from the Visualizer agent."""
    scene_type: str                        # "interactive_graph" | "3d_surface" | "comparison" | "animation_sequence"
    title: str
    elements: list[dict]                   # geometric elements with properties
    animations: list[dict]                 # animation steps with timing
    narration: str                         # 3B1B-style insight text
    interactive_params: list[dict]         # what user can adjust
    session_id: str = ""


class AssessorRequest(Model):
    """Request to the Assessor agent for contrastive challenge."""
    concept: str
    user_solution: str
    screen_content_type: str
    mastery_level: float
    user_id: str = "default"
    session_id: str = ""


class AssessorResponse(Model):
    """Contrastive challenge from the Assessor agent."""
    challenge: str                         # the question to present
    what_changes: str                      # what's different and why
    expected_insight: str                  # what they should realize
    connects_to: str                       # related concept for further study
    session_id: str = ""


class SessionReport(Model):
    """Report sent when a dialogue session closes."""
    session_id: str
    user_id: str
    concept: str
    turns_count: int
    final_comprehension: float
    observations: list[dict]               # BKT observations from the dialogue
    duration_seconds: float
    close_reason: str                      # "mastered" | "max_turns" | "timeout" | "user_closed"
