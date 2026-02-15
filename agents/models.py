"""
Message models for the Ambient Learning Agent System.

VLM context flows in → orchestrator routes by activity → agent generates exercise.
"""
from typing import Any, Optional
from uagents import Model


class VLMContext(Model):
    """Screen analysis from Gemini VLM — the core input to the system."""
    activity: str = ""              # what user is doing (reading, solving, reviewing)
    topic: str = ""                 # subject (e.g. "eigenvalues", "integration")
    subtopic: str = ""              # more specific sub-topic
    mode: str = ""                  # CONCEPTUAL | APPLIED | CONSOLIDATION
    content_type: str = "text"      # code | equation | text | diagram | mixed
    work_status: str = "unclear"    # correct | incorrect | incomplete | unclear
    stuck: bool = False
    error_description: Optional[str] = None
    notes: str = ""
    speech_transcript: Optional[str] = None  # what the user said out loud
    raw_vlm_text: str = ""          # full VLM output for agents to reference


class AgentRequest(Model):
    """Request sent from orchestrator to an agent."""
    vlm_context: VLMContext
    mastery: float = 0.0            # BKT mastery for the current topic (0-1)
    mastery_quality: str = "no_data"  # high | medium | low | no_data
    trigger_reason: str = ""         # why now: natural_pause | topic_transition | stuck | mode_change | fallback
    recent_observations: list[str] = []  # last few VLM summaries for context
    session_id: str = ""


class AgentResponse(Model):
    """Response from an agent back to the orchestrator / sidebar."""
    agent_type: str                 # "conceptual" | "applied" | "extension"
    content: str                    # the exercise, question, or prompt to show
    tool_used: str = "question"     # question | visualization | review | quiz
    topic: str = ""
    mastery: float = 0.0


class TimingSignal(Model):
    """Lightweight signal for the orchestrator to decide WHEN to prompt."""
    is_natural_pause: bool = False   # video paused, finished a problem, etc.
    is_transition: bool = False      # switched topic or activity
    seconds_on_same_content: float = 0.0
    just_finished_something: bool = False


# ─── Visualizer (tool returns latex | d3 | plotly | manim; UI embeds in sidebar) ───
class VisualizerRequest(Model):
    """Request to the visualizer: context from the agent that chose the visualization tool."""
    concept: str = ""
    subconcept: str = ""
    confusion_hypothesis: str = ""
    screen_context: str = ""
    student_question: str = ""
    session_id: str = ""


class VisualizerResponse(Model):
    """Structured response back to the orchestrator (optional)."""
    scene_type: str = ""
    title: str = ""
    elements: list = []
    animations: list = []
    narration: str = ""
    interactive_params: list = []
    session_id: str = ""


class AgentMessage(Model):
    """Message sent to the UI / sidebar (and optionally to orchestrator)."""
    content: str = ""
    content_type: str = "text"       # text | visualization
    agent_type: str = ""             # conceptual | visualizer | applied | ...
    session_id: str = ""
    tool_used: str = ""              # question | visualization | review | quiz
    metadata: Optional[dict] = None  # for visualization: tier, title, content/code/figure, narration
