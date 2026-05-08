"""
benchmark/metrics.py — All evaluation metrics for the comparative study.

Metrics computed:
  - ROUGE-L F1          (lexical overlap vs ideal answer)
  - BERTScore F1        (semantic similarity vs ideal answer)
  - Score Consistency   (std-dev across repeated runs)
  - Inter-Model Kappa   (Cohen's weighted kappa between model pairs)
  - Mean Latency        (wall-clock seconds per call)

Install dependencies first:
    pip install rouge-score bert-score scikit-learn
"""

import json
import time
from pathlib import Path


# ── ROUGE-L ────────────────────────────────────────────────────────────────────

def compute_rouge_l(hypothesis: str, reference: str) -> float:
    """Return ROUGE-L F1 between hypothesis and reference strings."""
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        score = scorer.score(reference, hypothesis)
        return round(score["rougeL"].fmeasure, 4)
    except ImportError:
        print("[metrics] rouge-score not installed. Run: pip install rouge-score")
        return _lcs_rouge_l(hypothesis, reference)


def _lcs_rouge_l(hyp: str, ref: str) -> float:
    """Pure-Python LCS fallback if rouge-score is unavailable."""
    a, b = hyp.lower().split(), ref.lower().split()
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return 0.0
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = dp[i-1][j-1] + 1 if a[i-1] == b[j-1] else max(dp[i-1][j], dp[i][j-1])
    lcs = dp[m][n]
    prec = lcs / m if m else 0
    rec  = lcs / n if n else 0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
    return round(f1, 4)


# ── BERTScore ──────────────────────────────────────────────────────────────────

def compute_bertscore(hypotheses: list[str], references: list[str]) -> list[float]:
    """
    Compute BERTScore F1 for a list of (hypothesis, reference) pairs.
    Returns a list of F1 floats, one per pair.
    Batched for efficiency.
    """
    try:
        from bert_score import score as bert_score_fn
        _, _, F1 = bert_score_fn(
            hypotheses, references,
            lang="en",
            model_type="bert-base-uncased",
            verbose=False,
        )
        return [round(f.item(), 4) for f in F1]
    except ImportError:
        print("[metrics] bert-score not installed. Run: pip install bert-score")
        # Fallback: use ROUGE-L as a proxy
        return [compute_rouge_l(h, r) for h, r in zip(hypotheses, references)]


# ── Latency ────────────────────────────────────────────────────────────────────

def timed_query(prompt: str, model_id: str, system_prompt: str = "") -> tuple[str, float]:
    """
    Call a model and return (response_text, elapsed_seconds).
    Import from model_registry inside to avoid circular imports.
    """
    from agents.model_registry import query
    t0 = time.time()
    response = query(prompt, model_id=model_id, system_prompt=system_prompt)
    elapsed = round(time.time() - t0, 2)
    return response, elapsed


# ── Score Consistency ──────────────────────────────────────────────────────────

def compute_consistency(scores: list[float]) -> float:
    """
    Standard deviation of scores across repeated runs.
    Lower = more consistent = better evaluator.
    """
    if len(scores) < 2:
        return 0.0
    mean = sum(scores) / len(scores)
    variance = sum((s - mean) ** 2 for s in scores) / len(scores)
    return round(variance ** 0.5, 4)


# ── Inter-Model Agreement (Cohen's Weighted Kappa) ────────────────────────────

def compute_kappa(scores_a: list[float], scores_b: list[float]) -> float:
    """
    Cohen's weighted kappa between two lists of overall scores.
    Scores are discretised into 3 bins: low(1-3), medium(4-7), high(8-10).
    Returns kappa in [-1, 1]; > 0.6 = substantial agreement.
    """
    try:
        from sklearn.metrics import cohen_kappa_score
        import numpy as np

        def bin_score(s):
            if s <= 3:   return 0   # low
            elif s <= 7: return 1   # medium
            else:        return 2   # high

        a_binned = [bin_score(s) for s in scores_a]
        b_binned = [bin_score(s) for s in scores_b]

        # cohen_kappa_score needs at least 2 classes present
        if len(set(a_binned)) < 2 and len(set(b_binned)) < 2:
            return 1.0 if a_binned == b_binned else 0.0

        kappa = cohen_kappa_score(a_binned, b_binned, weights="linear")
        return round(float(kappa), 4)

    except ImportError:
        print("[metrics] scikit-learn not installed. Run: pip install scikit-learn")
        return _simple_agreement(scores_a, scores_b)


def _simple_agreement(a: list[float], b: list[float]) -> float:
    """Fallback: proportion of matching bins."""
    def bin_score(s):
        if s <= 3: return 0
        elif s <= 7: return 1
        else: return 2
    matches = sum(1 for x, y in zip(a, b) if bin_score(x) == bin_score(y))
    return round(matches / len(a), 4) if a else 0.0


# ── Parse Evaluator Output ────────────────────────────────────────────────────

def parse_scores(raw: str) -> dict:
    """
    Extract numeric scores from evaluator agent output.
    Returns dict with keys: clarity, relevance, depth, structure, overall.
    """
    import re
    result = {"clarity": 5, "relevance": 5, "depth": 5, "structure": 5}

    for dim in ["clarity", "relevance", "depth", "structure"]:
        pattern = rf"(?:^|\n)\s*\**{dim}\**[:\s]+(\d+(?:\.\d+)?)"
        match = re.search(pattern, raw, re.IGNORECASE | re.MULTILINE)
        if match:
            result[dim] = min(10, max(1, round(float(match.group(1)))))

    scores = list(result.values())
    result["overall"] = round(sum(scores) / len(scores), 2)
    return result


# ── Aggregate Results Helper ──────────────────────────────────────────────────

def aggregate(values: list[float]) -> dict:
    """Return mean, min, max, std for a list of floats."""
    if not values:
        return {"mean": 0, "min": 0, "max": 0, "std": 0}
    mean = sum(values) / len(values)
    std  = (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5
    return {
        "mean": round(mean, 4),
        "min":  round(min(values), 4),
        "max":  round(max(values), 4),
        "std":  round(std, 4),
    }