"""
Message models for the Ambient Learning Agent System.

VLM context flows in → orchestrator routes by activity → agent generates exercise.
"""
from typing import Optional
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
