# Plato — 1-Minute Demo Script

**Duration:** 1 minute  
**Goal:** Plato knows when you’re stuck, watches with permission, decides when to prompt — no typing “I’m stuck” needed.

---

## Script (~1 min)

**"When you’re stuck, you usually have to stop and type ‘I’m stuck.’ We flipped that. This is Plato — an ambient learning companion."**

**"You don’t need to prompt when you’re stuck; Plato already knows. You give permission once, and Plato watches every tab and document you’re on. A vision-language model reads your screen in real time, and Plato decides when to prompt you: at natural pauses, when you’re stuck, or when you switch topics. It’s autonomous: no prompting required."**

**"Under the hood: we deployed three AI agents on Fetch.ai and two tools — voice call for oral concept checks with Plato, and visualization. We have a Conceptual agent when you’re reading or watching, an Applied agent when you’re solving problems or coding, and an Extension agent when you’re ready to stretch and connect ideas. Right now I’m working on a math problem, so our Applied agent is active. We use LaTeX, D3, Plotly, and Manim so you get active, interactive visuals — equations, diagrams, plots, and 3Blue1Brown-style animations — all driven by what’s on your screen."**

**[Demo beat]** Show one intervention (visualization or voice). *"I didn’t type anything. Plato saw the screen and prompted me."*

**"That’s Plato — when you’re stuck, it already knows."**

---

## One-liners (if you need to cut or reorder)

| Topic | Line |
|-------|------|
| No prompt | "You don’t need to prompt when you’re stuck — Plato already knows." |
| Permission | "When you give permission, Plato watches every tab or document you’re on." |
| Who prompts | "Plato decides when to prompt for you." |
| VLM | "A vision-language model interprets your screen in real time." |
| Stack | "Three agents on Fetch.ai, two tools: voice call and visualization. LaTeX, D3, Plotly, Manim for interactive visuals." |
| Oral | "Oral concept checks with Plato — talk through your reasoning." |

---

## Backup

- **VLM slow:** Run `python scripts/demo_scenario.py` to simulate stuck context and show agent responses.
- **Voice not working:** Say "Plato can also do voice concept checks — we’re showing the visualization here."
