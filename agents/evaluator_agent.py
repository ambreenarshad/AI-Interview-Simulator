"""
evaluator_agent.py — Evaluator Agent
Evaluates candidate answers on four dimensions using Ollama.
Returns structured scores with explanations.
"""

import re
from utils.llm import query_ollama


SYSTEM_PROMPT = """You are an expert interview evaluator. You assess candidate answers 
objectively and fairly, providing constructive numeric scores and brief explanations. 
You always respond in the exact structured format requested."""


def evaluate_answer(question: str, answer: str) -> dict:
    """
    Evaluate a candidate's answer on four dimensions.

    Returns a dict with:
        - clarity (int 1-10)
        - relevance (int 1-10)
        - depth (int 1-10)
        - structure (int 1-10)
        - overall (float, average of the four)
        - explanation (str)
        - raw_response (str, for debugging)
    """
    if not answer or answer.strip() == "":
        return _empty_answer_result()

    prompt = f"""Evaluate the following interview answer.

QUESTION: {question}

CANDIDATE'S ANSWER: {answer}

Score the answer on these FOUR criteria (each 1–10):

1. Clarity (1–10): Is the answer clear and easy to understand?
2. Relevance (1–10): Does it directly address the question asked?
3. Depth (1–10): Is it detailed and substantive, not vague or superficial?
4. Structure (1–10): Is it well-organized? (STAR method preferred: Situation, Task, Action, Result)

Respond in EXACTLY this format (no extra text before or after):

Clarity: X/10
Relevance: X/10
Depth: X/10
Structure: X/10
Overall: X/10
Explanation: <2–4 sentences explaining the scores and what the candidate did well or poorly>"""

    raw = query_ollama(prompt, SYSTEM_PROMPT)
    return _parse_evaluation(raw, question, answer)


def _parse_evaluation(raw: str, question: str, answer: str) -> dict:
    """Parse structured evaluation output from LLM."""
    result = {
        "clarity": 5,
        "relevance": 5,
        "depth": 5,
        "structure": 5,
        "overall": 5.0,
        "explanation": "Evaluation could not be fully parsed.",
        "raw_response": raw,
    }

    # Extract scores using regex — handles formats like "7/10", "7", "7.5"
    def extract_score(label: str) -> int:
        pattern = rf"{label}[:\s]+(\d+(?:\.\d+)?)\s*/?\s*10"
        match = re.search(pattern, raw, re.IGNORECASE)
        if match:
            return min(10, max(1, round(float(match.group(1)))))
        return 5  # default if not found

    result["clarity"] = extract_score("Clarity")
    result["relevance"] = extract_score("Relevance")
    result["depth"] = extract_score("Depth")
    result["structure"] = extract_score("Structure")

    # Extract explanation
    exp_match = re.search(r"Explanation[:\s]+(.+?)$", raw, re.IGNORECASE | re.DOTALL)
    if exp_match:
        result["explanation"] = exp_match.group(1).strip()

    # Compute overall average
    scores = [result["clarity"], result["relevance"], result["depth"], result["structure"]]
    result["overall"] = round(sum(scores) / len(scores), 2)

    return result


def _empty_answer_result() -> dict:
    """Return a default result for empty/no answer."""
    return {
        "clarity": 1,
        "relevance": 1,
        "depth": 1,
        "structure": 1,
        "overall": 1.0,
        "explanation": "No answer was provided. The candidate did not respond to this question.",
        "raw_response": "",
    }
