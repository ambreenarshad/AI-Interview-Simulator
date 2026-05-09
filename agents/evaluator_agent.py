"""
agents/evaluator_agent.py
=========================
Scores a candidate's interview answer on 4 dimensions using gemma2:2b.

Why gemma2 for evaluation?
──────────────────────────
Benchmark results across 30 samples × 3 consistency runs:

  Metric            llama3.2   gemma2    phi3    mistral
  ──────────────────────────────────────────────────────
  Consistency σ     0.3240    0.1551   0.7627   0.2965
  Latency (s)       100.5      66.9    179.1    206.9
  Overall score      7.75      8.24     5.81     8.52

gemma2 chosen because:
  1. Lowest consistency σ (0.155) — same answer gets same score reliably.
     Critical for fairness: a user re-submitting should get the same result.
  2. Fastest latency (66.9s) — reduces user wait time significantly.
  3. Strong overall score (8.24) — quality is not sacrificed for speed.

Evaluation approach: LLM-based scoring (NOT text similarity metrics).
Text similarity metrics (ROUGE, BLEU, BERTScore) require a reference answer
to compare against. In a live interview, no reference exists — the user's
answer is open-ended. LLM scoring assesses meaning, not surface text overlap.
"""

import re
import difflib
from utils.llm import query_evaluator   # gemma2:2b

SYSTEM_PROMPT = """You are a strict interview evaluator. You always respond in the EXACT format requested.
Never add extra text, explanations outside the format, or commentary."""


def evaluate_answer(question: str, answer: str) -> dict:
    """
    Score an interview answer on 4 dimensions using gemma2:2b.

    Parameters
    ----------
    question : the interview question asked
    answer   : the candidate's answer

    Returns
    -------
    dict with keys: clarity, relevance, depth, structure, overall (float),
                    explanation (str), raw_response (str)

    Scoring dimensions
    ------------------
    Clarity   : How clearly the answer is communicated (1-10)
    Relevance : How directly the answer addresses the question (1-10)
    Depth     : Level of insight, detail, and expertise shown (1-10)
    Structure : Organisation — logical flow, STAR method usage (1-10)
    Overall   : Mean of the 4 dimensions (float, 1-10)
    """
    if not answer or answer.strip() == "":
        return _empty_answer_result()

    q_norm = question.lower().strip()
    a_norm = answer.lower().strip()
    similarity = difflib.SequenceMatcher(None, q_norm, a_norm).ratio()
    if similarity > 0.75 or a_norm in q_norm or q_norm in a_norm:
        return {
            "clarity": 1, "relevance": 1, "depth": 1, "structure": 1,
            "overall": 1.0,
            "explanation": "The answer is identical or nearly identical to the question itself. Please provide a genuine response.",
            "raw_response": "",
        }

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

    # Uses gemma2:2b — chosen for lowest consistency σ (0.155) and fastest latency (66.9s)
    raw = query_evaluator(prompt, SYSTEM_PROMPT)
    return _parse_evaluation(raw)


def _parse_evaluation(raw: str) -> dict:
    result = {
        "clarity":     5,
        "relevance":   5,
        "depth":       5,
        "structure":   5,
        "overall":     5.0,
        "explanation": "Evaluation could not be fully parsed.",
        "raw_response": raw,
    }

    def extract(label: str) -> int:
        # Handles: "CLARITY: 7", "CLARITY: 7/10", "**CLARITY**: 7"
        pattern = rf"(?:^|\n)\s*\**{label}\**[:\s]+(\d+(?:\.\d+)?)"
        match = re.search(pattern, raw, re.IGNORECASE | re.MULTILINE)
        if match:
            return min(10, max(1, round(float(match.group(1)))))
        return 5

    result["clarity"]   = extract("CLARITY")
    result["relevance"] = extract("RELEVANCE")
    result["depth"]     = extract("DEPTH")
    result["structure"] = extract("STRUCTURE")

    exp_match = re.search(r"EXPLANATION[:\s]+(.+?)$", raw, re.IGNORECASE | re.DOTALL)
    if exp_match:
        result["explanation"] = exp_match.group(1).strip()

    scores = [result["clarity"], result["relevance"], result["depth"], result["structure"]]
    result["overall"] = round(sum(scores) / len(scores), 2)

    return result


def _empty_answer_result() -> dict:
    return {
        "clarity":   1,
        "relevance": 1,
        "depth":     1,
        "structure": 1,
        "overall":   1.0,
        "explanation": "No answer was provided.",
        "raw_response": "",
    }