"""
benchmark/run_benchmark.py — Runs all 3 models on the dataset and saves results.

What it produces:
  benchmark/results/raw_results.json      ← per-sample scores for all models
  benchmark/results/summary.json          ← aggregated metrics table (for paper)
  benchmark/results/consistency.json      ← per-model score std-dev
  benchmark/results/kappa.json            ← inter-model agreement pairs

Usage:
    python benchmark/run_benchmark.py

Runtime estimate (no GPU, consumer CPU):
    ~30 samples × 3 models × 2 calls (evaluator + feedback) × ~20s = ~36 minutes
    Reduce BENCHMARK_SAMPLES to 10 for a quick run (~12 minutes).
"""

import json
import sys
import time
from pathlib import Path

# ── Allow running from project root ───────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.model_registry import list_models, query
from benchmark.metrics import (
    compute_rouge_l,
    compute_bertscore,
    compute_consistency,
    compute_kappa,
    parse_scores,
    aggregate,
    timed_query,
)

# ── Config ─────────────────────────────────────────────────────────────────────
ROOT_DIR          = Path(__file__).resolve().parent.parent
DATASET_PATH      = ROOT_DIR / "dataset" / "combined_dataset.json"
RESULTS_DIR       = Path(__file__).resolve().parent / "results"
BENCHMARK_SAMPLES = 30      # set to 10 for a quick test run
REPEAT_RUNS       = 3       # for consistency measurement (same sample, 3 runs)
RANDOM_SEED       = 42

EVALUATOR_SYSTEM = """You are a strict interview evaluator. Respond ONLY in this exact format:
CLARITY: [1-10]
RELEVANCE: [1-10]
DEPTH: [1-10]
STRUCTURE: [1-10]
EXPLANATION: [2-3 sentences]"""

FEEDBACK_SYSTEM = """You are a professional interview coach. Respond ONLY in this exact format:
STRENGTHS: [what was done well]
WEAKNESSES: [what was lacking]
SUGGESTIONS: [actionable tips]
IMPROVED ANSWER: [better 3-5 sentence version]"""


# ── Prompts ────────────────────────────────────────────────────────────────────

def eval_prompt(question: str, answer: str) -> str:
    return f"""Evaluate this interview answer.

QUESTION: {question}
ANSWER: {answer}

Score each dimension 1-10. Respond in EXACTLY the format specified."""


def feedback_prompt(question: str, answer: str, scores: dict) -> str:
    scores_str = (f"Clarity {scores['clarity']}/10, Relevance {scores['relevance']}/10, "
                  f"Depth {scores['depth']}/10, Structure {scores['structure']}/10")
    return f"""Review this interview answer and give feedback.

QUESTION: {question}
ANSWER: {answer}
SCORES: {scores_str}

Respond in EXACTLY the format specified."""


# ── Sample selection ───────────────────────────────────────────────────────────

def load_benchmark_samples(n: int) -> list[dict]:
    """Load dataset and select a balanced sample of n entries."""
    import random
    random.seed(RANDOM_SEED)

    with open(DATASET_PATH) as f:
        dataset = json.load(f)

    # Stratified: pick proportionally across difficulty levels
    by_diff = {"easy": [], "medium": [], "hard": []}
    for entry in dataset:
        d = entry.get("difficulty", "medium")
        if d in by_diff:
            by_diff[d].append(entry)

    per_level = n // 3
    selected = []
    for diff, items in by_diff.items():
        selected.extend(random.sample(items, min(per_level, len(items))))

    # Top up to exactly n if rounding left gaps
    remaining = [e for e in dataset if e not in selected]
    random.shuffle(remaining)
    selected.extend(remaining[: n - len(selected)])

    return selected[:n]


# ── Main benchmark loop ────────────────────────────────────────────────────────

def run_benchmark():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if not DATASET_PATH.exists():
        print(f"[ERROR] Dataset not found at {DATASET_PATH}")
        print("        Run generate_dataset.py first.")
        sys.exit(1)

    samples  = load_benchmark_samples(BENCHMARK_SAMPLES)
    models   = list_models()          # ["llama", "gemma", "phi3"]

    print(f"\n{'='*65}")
    print(f"  AI Interview Simulator — Benchmark")
    print(f"  Models  : {', '.join(models)}")
    print(f"  Samples : {len(samples)}")
    print(f"  Repeats : {REPEAT_RUNS} (for consistency)")
    print(f"{'='*65}\n")

    # ── Storage ────────────────────────────────────────────────────────────────
    # raw_results[model_id] = list of per-sample dicts
    raw_results: dict[str, list[dict]] = {m: [] for m in models}

    # For consistency: store overall scores per model per sample (3 runs)
    consistency_scores: dict[str, list[list[float]]] = {m: [] for m in models}

    overall_count = len(samples) * len(models)
    done = 0
    t_start = time.time()

    for s_idx, sample in enumerate(samples):
        question     = sample["question"]
        ideal_answer = sample["ideal_answer"]
        difficulty   = sample.get("difficulty", "medium")
        role         = sample.get("role", "N/A")
        itype        = sample.get("type", "N/A")

        print(f"\n── Sample {s_idx+1}/{len(samples)} │ {role} │ {itype} │ {difficulty}")
        print(f"   Q: {question[:80]}...")

        for model_id in models:
            done += 1
            elapsed_total = time.time() - t_start
            rate = done / elapsed_total if elapsed_total > 0 else 0.1
            eta  = int((overall_count - done) / rate)
            print(f"\n   [{done}/{overall_count}] Model: {model_id:<6} │ ETA {eta//60}m {eta%60}s")

            # ── Evaluator call ─────────────────────────────────────────────
            ep = eval_prompt(question, ideal_answer)
            eval_raw, eval_latency = timed_query(ep, model_id, EVALUATOR_SYSTEM)
            scores = parse_scores(eval_raw)
            print(f"   Eval latency: {eval_latency}s │ Overall: {scores['overall']}")

            # ── Feedback call ──────────────────────────────────────────────
            fp = feedback_prompt(question, ideal_answer, scores)
            fb_raw, fb_latency = timed_query(fp, model_id, FEEDBACK_SYSTEM)

            # ── ROUGE-L against ideal answer ───────────────────────────────
            # Extract the IMPROVED ANSWER section for comparison
            import re
            imp_match = re.search(r"IMPROVED ANSWER[:\s]+(.+?)$", fb_raw,
                                   re.IGNORECASE | re.DOTALL)
            improved = imp_match.group(1).strip() if imp_match else fb_raw
            rouge_l  = compute_rouge_l(improved, ideal_answer)
            print(f"   Feedback latency: {fb_latency}s │ ROUGE-L: {rouge_l}")

            # ── Consistency: repeat evaluator REPEAT_RUNS times ───────────
            repeat_scores = [scores["overall"]]
            for r in range(REPEAT_RUNS - 1):
                r_raw, _ = timed_query(ep, model_id, EVALUATOR_SYSTEM)
                r_scores = parse_scores(r_raw)
                repeat_scores.append(r_scores["overall"])
            consistency_scores[model_id].append(repeat_scores)
            consistency = compute_consistency(repeat_scores)
            print(f"   Consistency σ: {consistency} (runs: {repeat_scores})")

            # ── Store result ───────────────────────────────────────────────
            raw_results[model_id].append({
                "sample_id":    sample.get("id", f"s{s_idx}"),
                "role":         role,
                "type":         itype,
                "difficulty":   difficulty,
                "question":     question,
                "ideal_answer": ideal_answer,
                "scores":       scores,
                "eval_latency": eval_latency,
                "fb_latency":   fb_latency,
                "rouge_l":      rouge_l,
                "consistency":  consistency,
                "repeat_scores": repeat_scores,
                "eval_raw":     eval_raw,
                "feedback_raw": fb_raw,
            })

        # Save progress after every sample
        _save_progress(raw_results, consistency_scores)

    # ── BERTScore (batched — much faster than per-sample) ─────────────────────
    print("\n\n── Computing BERTScore (batched)...")
    for model_id in models:
        hypotheses = []
        references = []
        for rec in raw_results[model_id]:
            imp = re.search(r"IMPROVED ANSWER[:\s]+(.+?)$",
                            rec["feedback_raw"], re.IGNORECASE | re.DOTALL)
            hypotheses.append(imp.group(1).strip() if imp else rec["feedback_raw"])
            references.append(rec["ideal_answer"])

        bert_scores = compute_bertscore(hypotheses, references)
        for rec, bs in zip(raw_results[model_id], bert_scores):
            rec["bertscore"] = bs
        print(f"   {model_id}: mean BERTScore = {aggregate(bert_scores)['mean']}")

    # ── Save raw results ───────────────────────────────────────────────────────
    raw_path = RESULTS_DIR / "raw_results.json"
    with open(raw_path, "w") as f:
        json.dump(raw_results, f, indent=2)
    print(f"\n✓ Raw results saved → {raw_path}")

    # ── Compute summary metrics ────────────────────────────────────────────────
    summary = {}
    for model_id in models:
        records = raw_results[model_id]
        rouge_vals  = [r["rouge_l"]   for r in records]
        bert_vals   = [r.get("bertscore", 0) for r in records]
        lat_vals    = [r["eval_latency"] + r["fb_latency"] for r in records]
        cons_vals   = [r["consistency"] for r in records]
        overall_vals = [r["scores"]["overall"] for r in records]

        # Per-difficulty BERTScore
        bert_by_diff = {}
        for diff in ["easy", "medium", "hard"]:
            vals = [r.get("bertscore", 0) for r in records if r["difficulty"] == diff]
            bert_by_diff[diff] = round(sum(vals) / len(vals), 4) if vals else 0

        summary[model_id] = {
            "rouge_l":        aggregate(rouge_vals),
            "bertscore":      aggregate(bert_vals),
            "latency_s":      aggregate(lat_vals),
            "consistency_std": aggregate(cons_vals),
            "overall_score":  aggregate(overall_vals),
            "bertscore_by_difficulty": bert_by_diff,
            "n_samples":      len(records),
        }

    sum_path = RESULTS_DIR / "summary.json"
    with open(sum_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Summary saved → {sum_path}")

    # ── Inter-model kappa ──────────────────────────────────────────────────────
    kappa_results = {}
    model_pairs = [
        ("llama", "gemma"),
        ("llama", "phi3"),
        ("gemma", "phi3"),
    ]
    for m1, m2 in model_pairs:
        if m1 not in raw_results or m2 not in raw_results:
            continue
        s1 = [r["scores"]["overall"] for r in raw_results[m1]]
        s2 = [r["scores"]["overall"] for r in raw_results[m2]]
        kappa = compute_kappa(s1, s2)
        kappa_results[f"{m1}_vs_{m2}"] = kappa
        print(f"   κ ({m1} vs {m2}) = {kappa}")

    kappa_path = RESULTS_DIR / "kappa.json"
    with open(kappa_path, "w") as f:
        json.dump(kappa_results, f, indent=2)
    print(f"✓ Kappa saved → {kappa_path}")

    # ── Print final table ──────────────────────────────────────────────────────
    _print_table(summary, kappa_results)

    total_time = int(time.time() - t_start)
    print(f"\n  Total runtime: {total_time//60}m {total_time%60}s")
    print(f"{'='*65}\n")


def _save_progress(raw_results, consistency_scores):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "_progress.json", "w") as f:
        json.dump({"raw": raw_results, "consistency": consistency_scores}, f)


def _print_table(summary: dict, kappa: dict):
    print(f"\n{'='*65}")
    print(f"  RESULTS SUMMARY")
    print(f"{'='*65}")
    header = f"{'Model':<10} {'ROUGE-L':>8} {'BERTScore':>10} {'Latency(s)':>11} {'Consist.σ':>10} {'κ(vs llama)':>12}"
    print(header)
    print("-" * 65)
    for model_id, m in summary.items():
        kappa_key = f"llama_vs_{model_id}" if model_id != "llama" else "—"
        kappa_val = kappa.get(kappa_key, kappa.get(f"{model_id}_vs_llama", "—"))
        kappa_str = f"{kappa_val:.4f}" if isinstance(kappa_val, float) else kappa_val
        print(
            f"{model_id:<10}"
            f"{m['rouge_l']['mean']:>8.4f}"
            f"{m['bertscore']['mean']:>10.4f}"
            f"{m['latency_s']['mean']:>11.2f}"
            f"{m['consistency_std']['mean']:>10.4f}"
            f"{kappa_str:>12}"
        )
    print(f"{'='*65}")

    print("\n  BERTScore by Difficulty:")
    print(f"  {'Model':<10} {'Easy':>8} {'Medium':>8} {'Hard':>8}")
    print("  " + "-" * 38)
    for model_id, m in summary.items():
        bd = m.get("bertscore_by_difficulty", {})
        print(f"  {model_id:<10} {bd.get('easy',0):>8.4f} {bd.get('medium',0):>8.4f} {bd.get('hard',0):>8.4f}")


if __name__ == "__main__":
    run_benchmark()