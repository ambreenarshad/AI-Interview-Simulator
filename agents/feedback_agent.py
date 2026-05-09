"""
feedback_agent.py — Generates actionable feedback for each answer
"""

import re
import json
from utils.llm import query_feedback

SYSTEM_PROMPT = """You are a professional interview coach. Output ONLY valid JSON. No markdown, no extra text."""

_GENERIC_PHRASES = (
    "good attempt", "could be more specific", "try using the star method",
    "not available", "be more specific", "use the star method",
)

def _is_generic(result: dict) -> bool:
    for key in ("strengths", "weaknesses", "suggestions", "improved_answer"):
        val = result.get(key, "").lower()
        if any(phrase in val for phrase in _GENERIC_PHRASES):
            return True
    return False


def _build_prompt(question: str, answer: str, scores_text: str, retry: bool = False) -> str:
    strict = (
        "\n\nCRITICAL: Do NOT write generic phrases like 'Good attempt', "
        "'Be more specific', or 'Try the STAR method'. "
        "Every sentence MUST reference specific words or ideas from the candidate's answer above."
        if retry else ""
    )
    return f"""You are reviewing a job interview answer. Give specific feedback that references what the candidate actually said.

QUESTION: {question}

CANDIDATE ANSWER: {answer}

SCORES: {scores_text}{strict}

Output ONLY a JSON object with these four keys. No other text:
{{
  "strengths": "2-3 specific things the candidate did well, referencing their actual words",
  "weaknesses": "2-3 specific gaps or missing elements in their answer",
  "suggestions": "2-3 concrete, actionable tips to improve this answer",
  "improved_answer": "A rewritten 3-5 sentence version using STAR method (Situation, Task, Action, Result)"
}}"""


def generate_feedback(question: str, answer: str, evaluation: dict) -> dict:
    if not answer or answer.strip() == "":
        return {
            "strengths": "No answer was provided.",
            "weaknesses": "The question was not attempted.",
            "suggestions": "Always attempt an answer. A partial answer is better than silence.",
            "improved_answer": "Structure your answer using STAR: Situation, Task, Action, Result.",
            "raw_response": "",
        }

    scores_text = (
        f"Clarity {evaluation['clarity']}/10, Relevance {evaluation['relevance']}/10, "
        f"Depth {evaluation['depth']}/10, Structure {evaluation['structure']}/10"
    )

    raw = query_feedback(_build_prompt(question, answer, scores_text), SYSTEM_PROMPT)
    result = _parse_feedback(raw)

    # If the response came back generic, retry once with a stricter prompt
    if _is_generic(result):
        raw2 = query_feedback(_build_prompt(question, answer, scores_text, retry=True), SYSTEM_PROMPT)
        result2 = _parse_feedback(raw2)
        if not _is_generic(result2):
            return result2

    return result


def _parse_feedback(raw: str) -> dict:
    result = {
        "strengths": "Good attempt.",
        "weaknesses": "Could be more specific.",
        "suggestions": "Try using the STAR method.",
        "improved_answer": "Not available.",
        "raw_response": raw,
    }

    if not raw or raw.startswith("[LLM Error]"):
        return result

    # --- Strategy 1: JSON ---
    # Find the outermost {...} block and try to parse it
    json_match = re.search(r'\{[\s\S]*\}', raw)
    if json_match:
        try:
            data = json.loads(json_match.group(0))
            for key in ("strengths", "weaknesses", "suggestions", "improved_answer"):
                val = data.get(key, "")
                if val and isinstance(val, str) and val.strip():
                    result[key] = val.strip()
            return result
        except (json.JSONDecodeError, AttributeError):
            pass

    # --- Strategy 2: labeled-section regex fallback ---
    def extract(label: str, stop_labels: list) -> str:
        escaped = re.escape(label)
        if stop_labels:
            stop = "|".join(re.escape(l) for l in stop_labels)
            pattern = rf"(?:^|\n)\s*\**{escaped}\**[:\s]+(.+?)(?=\n\s*\**(?:{stop})\**[:\s]|$)"
        else:
            pattern = rf"(?:^|\n)\s*\**{escaped}\**[:\s]+(.+?)$"
        m = re.search(pattern, raw, re.IGNORECASE | re.DOTALL)
        return m.group(1).strip() if m else ""

    s   = extract("STRENGTHS",       ["WEAKNESSES", "SUGGESTIONS", "IMPROVED ANSWER"])
    w   = extract("WEAKNESSES",      ["SUGGESTIONS", "IMPROVED ANSWER"])
    sug = extract("SUGGESTIONS",     ["IMPROVED ANSWER"])
    imp = extract("IMPROVED ANSWER", [])

    if s:   result["strengths"]      = s
    if w:   result["weaknesses"]     = w
    if sug: result["suggestions"]    = sug
    if imp: result["improved_answer"] = imp

    return result
