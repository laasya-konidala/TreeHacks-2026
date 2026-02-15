"""
Agentverse Registration Helper.

This script prints agent addresses and instructions for registering
on Agentverse. The actual registration happens automatically when you
run `python run.py` with AGENTVERSE_ENABLED=true.

Usage:
  1. Sign up at https://agentverse.ai (free)
  2. Run this script to see your agent addresses:
       python scripts/register_agentverse.py
  3. Set env vars and run the system:
       export AGENTVERSE_ENABLED=true
       export ANTHROPIC_API_KEY=sk-ant-...
       python run.py
  4. The agents auto-register on Agentverse and appear on ASI:One
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agents.config import (
    ORCHESTRATOR_SEED, ORCHESTRATOR_PORT,
    DEEP_DIVER_SEED, DEEP_DIVER_PORT,
    ASSESSOR_SEED, ASSESSOR_PORT,
    VISUALIZER_SEED, VISUALIZER_PORT,
    AGENTVERSE_ENABLED,
)


def main():
    # Import agents to get their addresses
    from uagents import Agent

    agents_info = [
        ("Orchestrator", ORCHESTRATOR_SEED, ORCHESTRATOR_PORT, "learning_orchestrator",
         "Central brain — detects confusion from behavioral signals, routes to specialists, tracks mastery via BKT"),
        ("Deep Diver", DEEP_DIVER_SEED, DEEP_DIVER_PORT, "concept_deep_diver",
         "Multi-turn Socratic dialogue for conceptual understanding"),
        ("Assessor", ASSESSOR_SEED, ASSESSOR_PORT, "contrastive_assessor",
         "Contrastive 'what-if' challenges to test understanding"),
        ("Visualizer", VISUALIZER_SEED, VISUALIZER_PORT, "math_visualizer",
         "3Blue1Brown-style visualization scene descriptions"),
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
        print("  ✓ Agents will auto-register on Agentverse when you run python run.py")
        print("  ✓ Chat Protocol included on orchestrator (discoverable on ASI:One)")
        print("  ✓ Payment Protocol included on orchestrator (monetization ready)")
    else:
        print("  → Agents will run locally only (no Agentverse registration)")
        print("  → To enable: export AGENTVERSE_ENABLED=true")

    print("\n" + "=" * 64)
    print("  HOW TO MAKE AGENTS DISCOVERABLE ON ASI:ONE")
    print("=" * 64)
    print("""
  Step 1: Sign up at https://agentverse.ai (free)

  Step 2: Set environment variables:
    export AGENTVERSE_ENABLED=true
    export GEMINI_API_KEY=your_gemini_api_key_here

  Step 3: Run the system:
    python run.py

  Step 4: The orchestrator agent auto-registers on Agentverse with:
    - Chat Protocol  → users on ASI:One can chat directly
    - Payment Protocol → monetization via ASI tokens
    - Description     → searchable on the Almanac

  Step 5: Go to ASI:One (asi1.ai) and search for your agent by
    name ("learning_orchestrator") or address to verify it's live.

  Note: The other 3 agents (deep_diver, assessor, visualizer) also
  register but they primarily receive messages from the orchestrator,
  not directly from ASI:One users.
""")

    # Check env vars
    print("=" * 64)
    print("  ENVIRONMENT CHECK")
    print("=" * 64)

    gemini_key = os.environ.get("GEMINI_API_KEY", "")
    key_status = "✓ SET" if gemini_key else "✗ NOT SET — Gemini will not work"
    print(f"  GEMINI_API_KEY: {key_status}")
    print(f"  AGENTVERSE_ENABLED: {'✓ true' if AGENTVERSE_ENABLED else '→ false (local only)'}")
    print()


if __name__ == "__main__":
    main()
