"""
Agentverse Registration Helper.

This script prints agent addresses and instructions for registering
on Agentverse. The actual registration happens automatically when you
run `python run.py` with AGENTVERSE_ENABLED=true and mailbox=True.

Usage:
  1. Sign up at https://agentverse.ai (free)
  2. Run this script to see your agent addresses:
       python scripts/register_agentverse.py
  3. Set env vars and run the system:
       export AGENTVERSE_ENABLED=true
       export GEMINI_API_KEY=...
       export ANTHROPIC_API_KEY=...
       python run.py
  4. The agents auto-register on Agentverse and appear on ASI:One
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agents.config import (
    ORCHESTRATOR_SEED, ORCHESTRATOR_PORT,
    CONCEPTUAL_SEED, CONCEPTUAL_PORT,
    APPLIED_SEED, APPLIED_PORT,
    EXTENSION_SEED, EXTENSION_PORT,
    MONITOR_SEED, MONITOR_PORT,
    AGENTVERSE_ENABLED,
)


def main():
    from uagents import Agent

    agents_info = [
        ("Orchestrator", ORCHESTRATOR_SEED, ORCHESTRATOR_PORT, "learning_orchestrator",
         "Central brain — ASI:One entry point with Chat + Payment protocols. "
         "Routes to conceptual/applied/extension agents, tracks mastery via BKT."),
        ("Conceptual", CONCEPTUAL_SEED, CONCEPTUAL_PORT, "conceptual_understanding",
         "Helps students build knowledge via contextual questions and visualizations."),
        ("Applied", APPLIED_SEED, APPLIED_PORT, "applied_problem_solving",
         "Scaffolds reasoning for active problem-solving without giving answers."),
        ("Extension", EXTENSION_SEED, EXTENSION_PORT, "extension_stretch",
         "Pushes students to make cross-topic connections and tackle stretch challenges."),
        ("Monitor", MONITOR_SEED, MONITOR_PORT, "metrics_monitor",
         "Sends metrics triggers to orchestrator using ASI-1 ChatProtocol."),
    ]

    print("\n" + "=" * 64)
    print("  AMBIENT LEARNING AGENT SYSTEM — Agent Addresses")
    print("=" * 64)

    for name, seed, port, agent_name, desc in agents_info:
        a = Agent(name=f"_tmp_{agent_name}", seed=seed)
        print(f"\n  {name}:")
        print(f"    Name:        {agent_name}")
        print(f"    Address:     {a.address}")
        print(f"    Port:        {port}")
        print(f"    Description: {desc}")

    print("\n" + "=" * 64)
    print("  AGENTVERSE STATUS")
    print("=" * 64)
    print(f"  AGENTVERSE_ENABLED = {AGENTVERSE_ENABLED}")

    if AGENTVERSE_ENABLED:
        print("  [ok] Agents will auto-register on Agentverse when you run: python run.py")
        print("  [ok] Chat Protocol on orchestrator (discoverable on ASI:One)")
        print("  [ok] Payment Protocol on orchestrator (FET monetization)")
    else:
        print("  --> Agents will run locally only (no Agentverse registration)")
        print("  --> To enable: export AGENTVERSE_ENABLED=true")

    print("\n" + "=" * 64)
    print("  HOW TO MAKE AGENTS DISCOVERABLE ON ASI:ONE")
    print("=" * 64)
    print("""
  Step 1: Sign up at https://agentverse.ai (free)

  Step 2: Set environment variables:
    export AGENTVERSE_ENABLED=true
    export GEMINI_API_KEY=your_gemini_api_key
    export ANTHROPIC_API_KEY=your_anthropic_api_key

  Step 3: Run the system:
    python run.py

  Step 4: The orchestrator auto-registers on Agentverse with:
    - Chat Protocol      -> users on ASI:One can chat directly
    - Payment Protocol   -> monetization via FET tokens
    - Agent description  -> searchable on the Almanac

  Step 5: Open the Local Agent Inspector URL from the terminal output
    to connect your agent to Agentverse (mailbox setup).

  Step 6: Go to ASI:One (asi1.ai), enable "Agents" toggle, and
    search for "ambient learning" or "learning orchestrator" to verify.

  Note: The sub-agents (conceptual, applied, extension) primarily
  receive messages from the orchestrator, not directly from ASI:One.
""")

    # Check env vars
    print("=" * 64)
    print("  ENVIRONMENT CHECK")
    print("=" * 64)

    for var, label in [
        ("GEMINI_API_KEY", "Gemini (screen analysis + chat)"),
        ("ANTHROPIC_API_KEY", "Claude (exercise generation)"),
        ("AGENTVERSE_ENABLED", "Agentverse registration"),
    ]:
        val = os.environ.get(var, "")
        if var == "AGENTVERSE_ENABLED":
            status = "[ok] true" if AGENTVERSE_ENABLED else "--> false (local only)"
        else:
            status = "[ok] SET" if val else "[!!] NOT SET"
        print(f"  {var}: {status}  ({label})")

    print()


if __name__ == "__main__":
    main()
