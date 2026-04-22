"""
feedback_agent.py — Feedback Agent
Generates rich, actionable feedback for each interview answer.
Includes strengths, weaknesses, improvement suggestions, and a model answer.
"""

from utils.llm import query_ollama


SYSTEM_PROMPT = """You are a supportive but honest interview coach with expertise in helping
candidates improve their interview performance. You provide specific, actionable feedback
and always frame criticism constructively."""


def generate_feedback(question: str, answer: str, evaluation: dict) -> dict:
    """
    Generate detailed feedback for a candidate's answer.

    Args:
        question: The interview question asked
        answer: The candidate's answer
        evaluation: The evaluation dict from evaluator_agent

    Returns:
        dict with keys:
            - strengths (str)
            - weaknesses (str)
            - suggestions (str)
            - improved_answer (str)
            - raw_response (str)
    """
    if not answer or answer.strip() == "":
        return {
            "strengths": "N/A — No answer was provided.",
            "weaknesses": "The candidate did not attempt to answer this question.",
            "suggestions": "Always attempt to answer, even if uncertain. A partial answer is better than silence.",
            "improved_answer": "I would approach this question by first thinking about relevant examples from my experience, then structuring my answer using the STAR method (Situation, Task, Action, Result).",
            "raw_response": "",
        }

    scores_text = (
        f"Clarity: {evaluation['clarity']}/10, "
        f"Relevance: {evaluation['relevance']}/10, "
        f"Depth: {evaluation['depth']}/10, "
        f"Structure: {evaluation['structure']}/10"
    )

    prompt = f"""You are reviewing a candidate's interview answer.

QUESTION: {question}

CANDIDATE'S ANSWER: {answer}

EVALUATION SCORES: {scores_text}
EVALUATOR'S NOTES: {evaluation.get('explanation', 'N/A')}

Provide feedback in EXACTLY this format:

STRENGTHS: <What the candidate did well — be specific, 2–3 points>
WEAKNESSES: <What was missing or could be better — be specific, 2–3 points>
SUGGESTIONS: <Concrete, actionable tips to improve this answer, 2–3 points>
IMPROVED ANSWER: <A concise, better version of the candidate's answer using the STAR method where applicable, 3–5 sentences>"""

    raw = query_ollama(prompt, SYSTEM_PROMPT)
    return _parse_feedback(raw)


def _parse_feedback(raw: str) -> dict:
    """Parse structured feedback from LLM response."""
    import re

    result = {
        "strengths": "Good attempt at answering the question.",
        "weaknesses": "Could provide more specific examples.",
        "suggestions": "Try using the STAR method for structured answers.",
        "improved_answer": "Not available.",
        "raw_response": raw,
    }

    def extract_section(label: str, next_labels: list[str]) -> str:
        """Extract text under a section label until the next label."""
        # Build pattern for "LABEL: content"
        next_pattern = "|".join(re.escape(l) for l in next_labels)
        pattern = rf"{re.escape(label)}[:\s]+(.+?)(?=(?:{next_pattern})[:\s]|$)"
        match = re.search(pattern, raw, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""

    s = extract_section("STRENGTHS", ["WEAKNESSES", "SUGGESTIONS", "IMPROVED ANSWER"])
    w = extract_section("WEAKNESSES", ["SUGGESTIONS", "IMPROVED ANSWER"])
    sug = extract_section("SUGGESTIONS", ["IMPROVED ANSWER"])
    imp = extract_section("IMPROVED ANSWER", [])

    if s:
        result["strengths"] = s
    if w:
        result["weaknesses"] = w
    if sug:
        result["suggestions"] = sug
    if imp:
        result["improved_answer"] = imp

    return result
