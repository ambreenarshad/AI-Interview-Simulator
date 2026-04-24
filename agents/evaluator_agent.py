"""
evaluator_agent.py — Scores answers on 4 dimensions with stable output format
"""

import re
from utils.llm import query_ollama

SYSTEM_PROMPT = """You are a strict interview evaluator. You always respond in the EXACT format requested.
Never add extra text, explanations outside the format, or commentary."""


def evaluate_answer(question: str, answer: str) -> dict:
    if not answer or answer.strip() == "":
        return _empty_answer_result()

    prompt = f"""Evaluate this interview answer strictly.

QUESTION: {question}

ANSWER: {answer}

Score each dimension 1-10. Respond in EXACTLY this format with no deviations:

CLARITY: [number]
RELEVANCE: [number]
DEPTH: [number]
STRUCTURE: [number]
EXPLANATION: [2-3 sentences about what was done well and what was lacking]

Only output those 5 lines. Nothing else."""

    raw = query_ollama(prompt, SYSTEM_PROMPT)
    return _parse_evaluation(raw)


def _parse_evaluation(raw: str) -> dict:
    result = {
        "clarity": 5, "relevance": 5, "depth": 5, "structure": 5,
        "overall": 5.0, "explanation": "Evaluation could not be fully parsed.",
        "raw_response": raw,
    }

    def extract(label: str) -> int:
        # Match "LABEL: 7" or "LABEL: 7/10" or "**LABEL**: 7"
        pattern = rf"(?:^|\n)\s*\**{label}\**[:\s]+(\d+(?:\.\d+)?)"
        match = re.search(pattern, raw, re.IGNORECASE | re.MULTILINE)
        if match:
            return min(10, max(1, round(float(match.group(1)))))
        return 5

    result["clarity"] = extract("CLARITY")
    result["relevance"] = extract("RELEVANCE")
    result["depth"] = extract("DEPTH")
    result["structure"] = extract("STRUCTURE")

    exp_match = re.search(r"EXPLANATION[:\s]+(.+?)$", raw, re.IGNORECASE | re.DOTALL)
    if exp_match:
        result["explanation"] = exp_match.group(1).strip()

    scores = [result["clarity"], result["relevance"], result["depth"], result["structure"]]
    result["overall"] = round(sum(scores) / len(scores), 2)

    return result


def _empty_answer_result() -> dict:
    return {
        "clarity": 1, "relevance": 1, "depth": 1, "structure": 1, "overall": 1.0,
        "explanation": "No answer was provided.",
        "raw_response": "",
    }
