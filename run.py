"""
Main entry point — starts Bureau (agents) + FastAPI server together.

Runs:
  - Orchestrator   (routes by VLM context, ASI:One chat + payment protocols)
  - Conceptual     (building knowledge)
  - Applied        (problem solving & scaffolding)
  - Extension      (stretch & connect)
  - Monitor        (metrics trigger via ASI-1 ChatProtocol)
"""
import logging
import threading

from dotenv import load_dotenv
load_dotenv()  # Load .env before anything reads os.environ

import uvicorn
from uagents import Bureau

from agents.config import (
    BACKEND_HOST, BACKEND_PORT, AGENTVERSE_ENABLED,
    ORCHESTRATOR_PORT, CONCEPTUAL_PORT, APPLIED_PORT, EXTENSION_PORT, MONITOR_PORT,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def start_api():
    """Start FastAPI server in background thread."""
    uvicorn.run(
        "input_pipeline.server:app",
        host=BACKEND_HOST,
        port=BACKEND_PORT,
        log_level="info",
    )


if __name__ == "__main__":
    # Start FastAPI in background thread
    api_thread = threading.Thread(target=start_api, daemon=True)
    api_thread.start()
    logger.info(f"API server starting on http://localhost:{BACKEND_PORT}")

    # Import agents
    from agents.orchestrator import orchestrator
    from agents.agent_conceptual import conceptual_agent
    from agents.agent_applied import applied_agent
    from agents.agent_extension import extension_agent
    from agents.monitor import monitor

    # Start all agents via Bureau
    bureau = Bureau()
    bureau.add(orchestrator)
    bureau.add(conceptual_agent)
    bureau.add(applied_agent)
    bureau.add(extension_agent)
    bureau.add(monitor)

    print("\n" + "=" * 64)
    print("  AMBIENT LEARNING AGENT SYSTEM")
    print("=" * 64)
    print(f"  Orchestrator:  {orchestrator.address}")
    print(f"    Port:        {ORCHESTRATOR_PORT}")
    print(f"    Protocols:   ChatProtocol (ASI:One), PaymentProtocol (FET)")
    print(f"  Conceptual:    {conceptual_agent.address}")
    print(f"    Port:        {CONCEPTUAL_PORT}")
    print(f"  Applied:       {applied_agent.address}")
    print(f"    Port:        {APPLIED_PORT}")
    print(f"  Extension:     {extension_agent.address}")
    print(f"    Port:        {EXTENSION_PORT}")
    print(f"  Monitor:       {monitor.address}")
    print(f"    Port:        {MONITOR_PORT}")
    print("=" * 64)
    print(f"  API Server:    http://localhost:{BACKEND_PORT}")
    print(f"  WebSocket:     ws://localhost:{BACKEND_PORT}/ws")
    print(f"  Health:        http://localhost:{BACKEND_PORT}/health")
    print("=" * 64)
    print(f"  Agentverse:    {'ENABLED — agents will register' if AGENTVERSE_ENABLED else 'disabled (local only)'}")
    if AGENTVERSE_ENABLED:
        print(f"  ASI:One:       Orchestrator discoverable via Chat Protocol")
        print(f"  Monetization:  Payment Protocol active (FET)")
    print("=" * 64 + "\n")

    bureau.run()
