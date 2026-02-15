"""
Main entry point — starts Bureau (agents) + FastAPI server together.

Currently runs:
  - Orchestrator (routes by VLM context)
  - Conceptual Understanding agent (building knowledge)
"""
import logging
import threading
import uvicorn
from uagents import Bureau

from agents.config import BACKEND_HOST, BACKEND_PORT

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

    # Start all agents via Bureau
    bureau = Bureau()
    bureau.add(orchestrator)
    bureau.add(conceptual_agent)

    print("\n" + "=" * 60)
    print("  AMBIENT LEARNING AGENT SYSTEM")
    print("=" * 60)
    print(f"  Orchestrator:  {orchestrator.address}")
    print(f"  Conceptual:    {conceptual_agent.address}")
    print(f"  API Server:    http://localhost:{BACKEND_PORT}")
    print(f"  WebSocket:     ws://localhost:{BACKEND_PORT}/ws")
    print(f"  Health:        http://localhost:{BACKEND_PORT}/health")
    print("=" * 60)
    print("  Agents: conceptual (applied + extension coming soon)")
    print("=" * 60 + "\n")

    bureau.run()
