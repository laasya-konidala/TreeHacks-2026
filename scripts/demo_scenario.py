"""
Demo scenario — tests the full backend pipeline WITHOUT any frontend.
Simulates a student working on linear regression who gets confused.

Usage:
  1. Start the system: python run.py
  2. In another terminal: python scripts/demo_scenario.py
"""
import asyncio
import json
import time

import httpx
import websockets

BACKEND_URL = "http://localhost:8080"
WS_URL = "ws://localhost:8080/ws"


async def listen_ws(messages: list):
    """Listen for agent responses via WebSocket."""
    try:
        async with websockets.connect(WS_URL) as ws:
            print("[WS] Connected to WebSocket")
            while True:
                try:
                    data = await asyncio.wait_for(ws.recv(), timeout=1.0)
                    msg = json.loads(data)
                    messages.append(msg)
                    print(f"\n{'='*50}")
                    print(f"[AGENT: {msg.get('agent_type', '?')}] ({msg.get('content_type', '?')})")
                    print(f"State: {msg.get('dialogue_state', 'n/a')}")
                    print(f"Content: {msg.get('content', '')[:200]}")
                    if msg.get('metadata'):
                        print(f"Metadata: {json.dumps(msg['metadata'], indent=2)[:200]}")
                    print(f"{'='*50}\n")
                except asyncio.TimeoutError:
                    continue
    except Exception as e:
        print(f"[WS] Disconnected: {e}")


async def main():
    print("\n" + "=" * 60)
    print("  DEMO: Simulating a confused student")
    print("=" * 60)

    # Check health
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(f"{BACKEND_URL}/health")
            print(f"\n[Health] {resp.json()}")
        except Exception:
            print("\n[ERROR] Backend not running! Start with: python run.py")
            return

    # Start WebSocket listener
    messages = []
    ws_task = asyncio.create_task(listen_ws(messages))

    await asyncio.sleep(1)  # Let WS connect

    async with httpx.AsyncClient() as client:

        # ─── Step 1: Student working normally ───
        print("\n--- Step 1: Student working normally (no confusion) ---")
        await client.post(f"{BACKEND_URL}/context", json={
            "screen_content": "import numpy as np\nX = np.array([[1, 1], [1, 2], [1, 3]])\ny = np.array([1, 2, 3])\ntheta = np.linalg.inv(X.T @ X) @ X.T @ y",
            "screen_content_type": "code",
            "detected_topic": "linear_regression",
            "detected_subtopic": "normal_equation",
            "typing_speed_ratio": 1.1,
            "deletion_rate": 1.0,
            "pause_duration": 2.0,
            "scroll_back_count": 0,
            "user_id": "demo_student",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })
        print("[Sent] Normal working context")
        await asyncio.sleep(5)  # Wait for orchestrator to poll

        # ─── Step 2: Student starts struggling ───
        print("\n--- Step 2: Student starts struggling ---")
        await client.post(f"{BACKEND_URL}/context", json={
            "screen_content": "# Why doesn't this work?\n# theta = np.linalg.inv(X.T @ X) @ X.T @ y\n# Getting singular matrix error\n# X.T @ X is not invertible??\nprint(np.linalg.det(X.T @ X))  # 0.0 ???",
            "screen_content_type": "code",
            "detected_topic": "linear_regression",
            "detected_subtopic": "normal_equation",
            "typing_speed_ratio": 0.3,
            "deletion_rate": 8.0,
            "pause_duration": 25.0,
            "scroll_back_count": 4,
            "user_id": "demo_student",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })
        print("[Sent] Struggling context (should trigger intervention!)")
        await asyncio.sleep(8)  # Wait for agent response

        # ─── Step 3: Explicit help request ───
        print("\n--- Step 3: Explicit help request ---")
        await client.post(f"{BACKEND_URL}/touch", json={
            "message": "Why is my matrix not invertible?"
        })
        print("[Sent] Explicit help request")
        await asyncio.sleep(8)

        # ─── Step 4: User replies to deep diver ───
        if messages:
            last_msg = messages[-1]
            session_id = last_msg.get("session_id", "")
            if session_id:
                print(f"\n--- Step 4: User replies in session {session_id} ---")
                await client.post(f"{BACKEND_URL}/reply", json={
                    "message": "I think it's because the columns are linearly dependent? But I don't understand why that matters for the normal equation.",
                    "session_id": session_id,
                    "user_id": "demo_student",
                })
                print("[Sent] User reply")
                await asyncio.sleep(8)

        # ─── Step 5: Visual request ───
        print("\n--- Step 5: Visual help request ---")
        await client.post(f"{BACKEND_URL}/context", json={
            "screen_content": "Loss function surface for gradient descent",
            "screen_content_type": "equation",
            "detected_topic": "gradient_descent",
            "detected_subtopic": "loss_landscape",
            "typing_speed_ratio": 0.4,
            "deletion_rate": 5.0,
            "pause_duration": 15.0,
            "scroll_back_count": 3,
            "user_touched_agent": True,
            "user_message": "Can you visualize this for me?",
            "user_id": "demo_student",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })
        print("[Sent] Visual help request")
        await asyncio.sleep(8)

    # Summary
    print("\n" + "=" * 60)
    print(f"  DEMO COMPLETE — Received {len(messages)} agent messages")
    print("=" * 60)
    for i, msg in enumerate(messages):
        print(f"  {i+1}. [{msg.get('agent_type')}] {msg.get('content_type')}: {msg.get('content', '')[:80]}...")

    ws_task.cancel()
    try:
        await ws_task
    except asyncio.CancelledError:
        pass


if __name__ == "__main__":
    asyncio.run(main())
