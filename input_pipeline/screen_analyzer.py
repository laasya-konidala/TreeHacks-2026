"""
Screen analyzer — Gemini Vision API screenshot analysis.
Analyzes screenshots to determine what concept the student is working on
and whether their work is correct.
"""
import json
import logging
from typing import Optional

from google import genai
from google.genai import types

from agents.config import GEMINI_MODEL, GEMINI_API_KEY

logger = logging.getLogger(__name__)

client = genai.Client(api_key=GEMINI_API_KEY)

ANALYSIS_PROMPT = """Analyze this screenshot of a student's work.
You are an assessment engine, NOT a tutor.

Determine:
1. What specific concept/topic is the student working on?
2. Is their current work CORRECT, INCORRECT, or INCOMPLETE?
3. If incorrect, what is the SPECIFIC error?
4. What concepts does this demonstrate understanding of?
5. What concepts does this demonstrate confusion about?

Return ONLY valid JSON (no markdown fences):
{
  "concept_id": "string - short snake_case identifier",
  "subconcept": "string - more specific topic",
  "work_status": "correct" | "incorrect" | "incomplete" | "unclear",
  "error_type": null | "procedural" | "conceptual" | "notational" | "arithmetic",
  "specific_error": null | "description of the error",
  "demonstrates_understanding_of": ["concept1", "concept2"],
  "demonstrates_confusion_about": ["concept3"],
  "confidence": 0.0 to 1.0
}"""


async def analyze_screenshot(screenshot_b64: str) -> Optional[dict]:
    """
    Analyze a screenshot using Gemini Vision API.

    Args:
        screenshot_b64: Base64-encoded PNG image

    Returns:
        Structured analysis dict or None on failure
    """
    if not screenshot_b64:
        return None

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_text(text=ANALYSIS_PROMPT),
                        types.Part.from_bytes(
                            data=__import__("base64").b64decode(screenshot_b64),
                            mime_type="image/png",
                        ),
                    ],
                ),
            ],
            config=types.GenerateContentConfig(
                max_output_tokens=500,
            ),
        )

        text = response.text or ""

        # Strip markdown fences if present
        text = text.replace("```json", "").replace("```", "").strip()

        result = json.loads(text)
        logger.info(f"Screen analysis: concept={result.get('concept_id')}, "
                     f"status={result.get('work_status')}")
        return result

    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse screen analysis JSON: {e}")
        return None
    except Exception as e:
        logger.warning(f"Screen analysis failed: {e}")
        return None
