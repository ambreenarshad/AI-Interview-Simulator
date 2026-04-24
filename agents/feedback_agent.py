"""
feedback_agent.py — Generates actionable feedback for each answer
"""

import re
from utils.llm import query_ollama

SYSTEM_PROMPT = """You are a professional interview coach. You give specific, actionable feedback.
Always respond in the EXACT format requested — no extra text."""


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

    prompt = f"""Review this interview answer and give feedback.

QUESTION: {question}
ANSWER: {answer}
SCORES: {scores_text}

Respond in EXACTLY this format:

STRENGTHS: [2-3 specific things done well]
WEAKNESSES: [2-3 specific things that were lacking]
SUGGESTIONS: [2-3 concrete actionable tips]
IMPROVED ANSWER: [A better 3-5 sentence version using STAR method]

Only output those 4 labeled sections. Nothing else."""

    raw = query_ollama(prompt, SYSTEM_PROMPT)
    return _parse_feedback(raw)


def _parse_feedback(raw: str) -> dict:
    result = {
        "strengths": "Good attempt.",
        "weaknesses": "Could be more specific.",
        "suggestions": "Try using the STAR method.",
        "improved_answer": "Not available.",
        "raw_response": raw,
    }

    def extract(label: str, stop_labels: list[str]) -> str:
        stop = "|".join(re.escape(l) for l in stop_labels)
        pattern = rf"(?:^|\n)\s*{re.escape(label)}[:\s]+(.+?)(?=(?:\n\s*(?:{stop})[:\s])|$)"
        match = re.search(pattern, raw, re.IGNORECASE | re.DOTALL)
        return match.group(1).strip() if match else ""

    s = extract("STRENGTHS", ["WEAKNESSES", "SUGGESTIONS", "IMPROVED ANSWER"])
    w = extract("WEAKNESSES", ["SUGGESTIONS", "IMPROVED ANSWER"])
    sug = extract("SUGGESTIONS", ["IMPROVED ANSWER"])
    imp = extract("IMPROVED ANSWER", [])

    if s: result["strengths"] = s
    if w: result["weaknesses"] = w
    if sug: result["suggestions"] = sug
    if imp: result["improved_answer"] = imp

    return result
