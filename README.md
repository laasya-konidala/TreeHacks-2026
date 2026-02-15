# Ambient Learning Agent System

![tag:innovationlab](https://img.shields.io/badge/innovationlab-3D8BD3)
![tag:hackathon](https://img.shields.io/badge/hackathon-5F43F1)

A multi-agent AI tutoring system that **observes what you're studying** and **helps you learn** through contextual questions, visualizations, and guided problem-solving — all without interrupting your flow.

Built on [Fetch.ai](https://fetch.ai) uAgents with ASI:One Chat Protocol and Payment Protocol for discoverability and monetization on [Agentverse](https://agentverse.ai).

## Agents

| Agent | Name | Description |
|-------|------|-------------|
| **Orchestrator** | `learning_orchestrator` | Central brain — ASI:One entry point. Detects when to prompt, routes to specialist agents, tracks mastery via BKT. Includes Chat + Payment protocols. |
| **Conceptual** | `conceptual_understanding` | Helps students build knowledge via contextual questions and visualizations when watching videos or reading notes. |
| **Applied** | `applied_problem_solving` | Scaffolds reasoning for active problem-solving — guides without giving answers. |
| **Extension** | `extension_stretch` | Pushes students to make cross-topic connections and tackle stretch challenges. |
| **Monitor** | `metrics_monitor` | Sends metrics triggers to orchestrator using ASI-1 ChatProtocol. |

## Architecture

```
ASI:One User
    │ ChatMessage (chat_protocol_spec)
    ▼
┌─────────────────────────────────────────────┐
│  Orchestrator (learning_orchestrator)        │
│  ├── Chat Protocol   (ASI:One discoverable) │
│  ├── Payment Protocol (FET monetization)     │
│  ├── BKT Learner Model                      │
│  └── Timing Logic (when to prompt)          │
└──────────┬──────────┬──────────┬────────────┘
           │          │          │
    ┌──────▼──┐ ┌─────▼────┐ ┌──▼──────────┐
    │Conceptual│ │ Applied  │ │  Extension  │
    │ Agent    │ │  Agent   │ │   Agent     │
    └─────────┘ └──────────┘ └─────────────┘
           │          │          │
           └──────────┴──────────┘
                      │
                  Claude LLM
              (exercise generation)
```

**Data flow:**
1. Electron desktop overlay captures screen → Gemini VLM analysis → FastAPI backend
2. Orchestrator polls backend, updates BKT mastery model, detects natural prompt moments
3. Routes to conceptual/applied/extension agent based on activity mode
4. Agent generates contextual exercise via Claude → response sent to sidebar via WebSocket

## Setup

### Prerequisites

- Python 3.10+
- Node.js 18+ (for Electron overlay)
- API keys: Gemini, Anthropic (Claude)

### Install

```bash
# Python dependencies
pip install -r requirements.txt

# Electron overlay
npm install
```

### Environment Variables

```bash
export GEMINI_API_KEY=your_gemini_key
export ANTHROPIC_API_KEY=your_anthropic_key

# To enable Agentverse registration:
export AGENTVERSE_ENABLED=true

# For FET payment (testnet by default):
export FET_USE_TESTNET=true
```

### Run

```bash
# Start all agents + API server
python run.py

# In a separate terminal, start the Electron overlay
npm start
```

### Agent Addresses

To see all agent addresses and registration status:

```bash
python scripts/register_agentverse.py
```

## Agentverse Deployment

1. Set `AGENTVERSE_ENABLED=true` and run `python run.py`
2. The orchestrator auto-registers on Agentverse with `mailbox=True`
3. Open the **Local Agent Inspector** URL from terminal output to connect
4. The Chat Protocol manifest is published — ASI:One can discover the agent
5. Go to [ASI:One](https://asi1.ai), enable "Agents" toggle, and search for your agent

## Monetization

The Payment Protocol supports FET token payments on the Fetch.ai blockchain:

- **Free tier**: 3 interventions/day, behavioral detection only
- **Premium tier**: Unlimited interventions + screen analysis + multi-turn dialogue (0.1 FET)
- **Per-mastery tier**: Charge on concept mastery milestones

## Tech Stack

- **Agents**: Fetch.ai uAgents framework
- **LLMs**: Claude (exercise generation), Gemini (screen analysis, chat)
- **Frontend**: Electron desktop overlay + Chrome extension
- **Backend**: FastAPI + WebSocket
- **Learning Model**: Confidence-Weighted Bayesian Knowledge Tracing (BKT)

## Extra Resources

- Chrome extension for behavioral signals: `learning-companion-extension/`
- Demo scenario: `scripts/demo_scenario.py`
- Vision pipeline test: `scripts/test_vision_pipeline.py`
