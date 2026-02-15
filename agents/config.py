"""
Configuration for the Ambient Learning Agent System.
Seeds, API keys, thresholds, and agent addresses.
"""
import os

# ─── Agent Seeds (deterministic addresses) ───
ORCHESTRATOR_SEED = "ambient_learning_orchestrator_seed_2026"
CONCEPTUAL_SEED = "ambient_learning_conceptual_seed_2026"
APPLIED_SEED = "ambient_learning_applied_seed_2026"
EXTENSION_SEED = "ambient_learning_extension_seed_2026"
MONITOR_SEED = "ambient_learning_monitor_seed_2026"
VISUALIZER_SEED = "ambient_learning_visualizer_seed_2026"

# ─── Agent Ports ───
ORCHESTRATOR_PORT = 8000
CONCEPTUAL_PORT = 8002
APPLIED_PORT = 8003
EXTENSION_PORT = 8004
MONITOR_PORT = 8005
VISUALIZER_PORT = 8006

# ─── API Configuration ───
BACKEND_HOST = "0.0.0.0"
BACKEND_PORT = 8080
BACKEND_URL = f"http://localhost:{BACKEND_PORT}"

# ─── Gemini (VLM — screen analysis in Electron) ───
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-2.0-flash"

# ─── Claude (Agent LLM — exercise generation) ───
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL = "claude-sonnet-4-5"

# ─── Agentverse ───
AGENTVERSE_ENABLED = os.environ.get("AGENTVERSE_ENABLED", "false").lower() == "true"
AGENTVERSE_URL = os.environ.get("AGENTVERSE_URL", "https://agentverse.ai")
AGENTVERSE_API_KEY = os.environ.get("AGENTVERSE_API_KEY", "")

# ─── FET Payment ───
FET_USE_TESTNET = os.environ.get("FET_USE_TESTNET", "true").lower() == "true"

# ─── Prompting Timing ───
MIN_SECONDS_BETWEEN_PROMPTS = 30    # don't spam
NATURAL_PAUSE_THRESHOLD = 15        # seconds before considering a prompt

# ─── BKT Defaults ───
BKT_DEFAULT_PRIOR = 0.3
BKT_P_LEARN = 0.1
BKT_P_GUESS = 0.25
BKT_P_SLIP = 0.1
BKT_MASTERY_THRESHOLD = 0.85
