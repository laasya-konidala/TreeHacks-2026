"""
Programmatic Agentverse Registration.

Registers agents on Agentverse without needing the Local Agent Inspector.
Requires AGENTVERSE_API_KEY from agentverse.ai (Settings → API Keys).

Usage:
    export AGENTVERSE_API_KEY=your_key_here
    python scripts/setup_agentverse.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv()

from uagents_core.utils.registration import (
    register_chat_agent,
    RegistrationRequestCredentials,
)

from agents.config import (
    ORCHESTRATOR_SEED, ORCHESTRATOR_PORT,
    CONCEPTUAL_SEED, CONCEPTUAL_PORT,
    APPLIED_SEED, APPLIED_PORT,
    EXTENSION_SEED, EXTENSION_PORT,
    MONITOR_SEED, MONITOR_PORT,
)

AGENTVERSE_API_KEY = os.environ.get("AGENTVERSE_API_KEY", "")

if not AGENTVERSE_API_KEY:
    print("ERROR: AGENTVERSE_API_KEY not set!")
    print("  1. Go to agentverse.ai → Settings → API Keys")
    print("  2. Create a new API key")
    print("  3. Add to .env: AGENTVERSE_API_KEY=your_key")
    print("  4. Re-run this script")
    sys.exit(1)


# Agents to register — name, seed, port, description
AGENTS = [
    (
        "learning_orchestrator",
        ORCHESTRATOR_SEED,
        ORCHESTRATOR_PORT,
        "Ambient Learning Orchestrator — an AI tutoring agent that observes what "
        "you're studying, detects when you need help, and provides contextual "
        "questions, visualizations, and guided problem-solving. Ask about any "
        "topic — math, science, programming. Includes Chat and Payment protocols.",
    ),
    (
        "conceptual_understanding",
        CONCEPTUAL_SEED,
        CONCEPTUAL_PORT,
        "Conceptual Understanding agent — helps students build knowledge via "
        "contextual questions and visualizations.",
    ),
    (
        "applied_problem_solving",
        APPLIED_SEED,
        APPLIED_PORT,
        "Applied problem-solving agent — scaffolds reasoning without giving answers.",
    ),
    (
        "extension_stretch",
        EXTENSION_SEED,
        EXTENSION_PORT,
        "Extension agent — pushes students to make cross-topic connections.",
    ),
    (
        "metrics_monitor",
        MONITOR_SEED,
        MONITOR_PORT,
        "Metrics monitor — triggers learning interventions via ChatProtocol.",
    ),
]


def main():
    print("\n" + "=" * 64)
    print("  AGENTVERSE REGISTRATION")
    print("=" * 64)

    success_count = 0
    for name, seed, port, description in AGENTS:
        endpoint = f"http://127.0.0.1:{port}/submit"
        print(f"\n  Registering: {name}")
        print(f"    Seed: {seed[:30]}...")
        print(f"    Endpoint: {endpoint}")

        try:
            register_chat_agent(
                name,
                endpoint,
                active=True,
                credentials=RegistrationRequestCredentials(
                    agentverse_api_key=AGENTVERSE_API_KEY,
                    agent_seed_phrase=seed,
                ),
            )
            print(f"    [OK] Registered successfully!")
            success_count += 1
        except Exception as e:
            print(f"    [FAIL] {e}")

    print(f"\n{'=' * 64}")
    print(f"  Registered {success_count}/{len(AGENTS)} agents on Agentverse")
    print(f"{'=' * 64}")

    if success_count == len(AGENTS):
        print("""
  All agents registered! Next steps:

  1. Run the agents:
       python run.py

  2. The mailbox connections should now work (no inspector needed).

  3. Test on ASI:One:
       - Go to asi1.ai
       - Toggle "Agents" on
       - Ask: "help me study calculus" or "explain eigenvalues"
""")
    else:
        print("\n  Some registrations failed. Check the errors above.")
        print("  Common issues:")
        print("    - Invalid API key")
        print("    - Network connectivity")
        print("    - Agent seed already registered under different account")


if __name__ == "__main__":
    main()
