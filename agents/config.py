"""
Configuration for the Ambient Learning Agent System.
Seeds, API keys, thresholds, and agent addresses.
"""
import os

# ─── Agent Seeds (deterministic addresses) ───
ORCHESTRATOR_SEED = "ambient_learning_orchestrator_seed_2026"
DEEP_DIVER_SEED = "ambient_learning_deep_diver_seed_2026"
ASSESSOR_SEED = "ambient_learning_assessor_seed_2026"
VISUALIZER_SEED = "ambient_learning_visualizer_seed_2026"

# ─── Agent Ports ───
ORCHESTRATOR_PORT = 8000
DEEP_DIVER_PORT = 8002
ASSESSOR_PORT = 8003
VISUALIZER_PORT = 8004

# ─── API Configuration ───
BACKEND_HOST = "0.0.0.0"
BACKEND_PORT = 8080
BACKEND_URL = f"http://localhost:{BACKEND_PORT}"

# ─── Gemini (Python backend) ───
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-2.5-flash"

# ─── Agentverse ───
# Set to True when you want agents to register on Agentverse and be
# discoverable on ASI:One. Requires signing up at agentverse.ai first.
# When True, agents connect via mailbox so they can receive messages
# from the Fetch.ai network (not just localhost).
AGENTVERSE_ENABLED = os.environ.get("AGENTVERSE_ENABLED", "false").lower() == "true"
AGENTVERSE_URL = os.environ.get("AGENTVERSE_URL", "https://agentverse.ai")

# ─── Confusion Detection Thresholds ───
CONFUSION_THRESHOLD = 0.42
INTERVENTION_COOLDOWN_SECONDS = 30
SCREEN_ANALYSIS_INTERVAL_SECONDS = 10

# ─── BKT Defaults ───
BKT_DEFAULT_PRIOR = 0.3
BKT_P_LEARN = 0.1
BKT_P_GUESS = 0.25
BKT_P_SLIP = 0.1
BKT_MASTERY_THRESHOLD = 0.85

# ─── Dialogue Session ───
MAX_DIALOGUE_TURNS = 10
MAX_DIALOGUE_DURATION_SECONDS = 300
CLOSING_COMPREHENSION_STREAK = 3
CLOSING_COMPREHENSION_THRESHOLD = 0.7

# ─── Signal Weights (confusion detector) ───
WEIGHT_TYPING = 0.15
WEIGHT_DELETION = 0.10
WEIGHT_PAUSE = 0.15
WEIGHT_REREAD = 0.10
WEIGHT_VERBAL = 0.15
WEIGHT_TOUCH = 0.10
WEIGHT_GEMINI = 0.25          # Gemini VLM analysis (strongest signal)
