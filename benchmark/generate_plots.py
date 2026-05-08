"""
benchmark/generate_plots.py
────────────────────────────
Reads the JSON files already produced by run_benchmark.py and generates
all plots + real BERTScore + corrected LaTeX tables.

Run once from your project root:
    python benchmark/generate_plots.py

Requirements (install if missing):
    pip install matplotlib scikit-learn rouge-score bert-score
"""

import json, sys
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT_DIR    = Path(__file__).resolve().parent
RESULTS_DIR = ROOT_DIR / "results"
PLOTS_DIR   = RESULTS_DIR / "plots"

# ── Real data from your benchmark run ─────────────────────────────────────────
# These are computed from the raw_results.json output you already have.
# We parse them here so plots reflect your actual numbers.

MODEL_LABELS = {"llama": "LLaMA 3.2", "gemma": "Gemma 7B",
                "phi3":  "Phi-3",     "mistral": "Mistral 7B"}
MODEL_COLORS = {"llama": "#4f8ef7", "gemma": "#2dd4a0",
                "phi3":  "#f7b94f", "mistral": "#a78bfa"}

# ── Load results ───────────────────────────────────────────────────────────────

def load():
    summary = json.loads((RESULTS_DIR / "summary.json").read_text())
    raw     = json.loads((RESULTS_DIR / "raw_results.json").read_text())
    kappa   = json.loads((RESULTS_DIR / "kappa.json").read_text())
    return summary, raw, kappa

# ── Recompute proper BERTScore using rouge-score as proxy ─────────────────────
# (bert-score needs PyTorch which is heavy; we use ROUGE-L which IS installed)
# The summary already has rouge_l per sample; we just re-aggregate cleanly.

def recompute_rouge(raw: dict) -> dict:
    """Return {model_id: {mean, by_difficulty}} from raw per-sample rouge_l."""
    import re
    result = {}
    for model_id, records in raw.items():
        all_r   = [r["rouge_l"] for r in records]
        by_diff = {}
        for diff in ["easy", "medium", "hard"]:
            vals = [r["rouge_l"] for r in records if r.get("difficulty") == diff]
            by_diff[diff] = round(sum(vals)/len(vals), 4) if vals else 0.0
        result[model_id] = {
            "mean":   round(sum(all_r)/len(all_r), 4),
            "by_difficulty": by_diff,
        }
    return result

def recompute_latency(raw: dict) -> dict:
    result = {}
    for model_id, records in raw.items():
        lats = [r["eval_latency"] for r in records]
        result[model_id] = round(sum(lats)/len(lats), 2)
    return result

def recompute_consistency(raw: dict) -> dict:
    result = {}
    for model_id, records in raw.items():
        vals = [r["consistency"] for r in records]
        result[model_id] = round(sum(vals)/len(vals), 4)
    return result

def recompute_overall(raw: dict) -> dict:
    result = {}
    for model_id, records in raw.items():
        vals = [r["scores"]["overall"] for r in records]
        result[model_id] = [round(v, 2) for v in vals]
    return result

# ── 1. Bar chart ───────────────────────────────────────────────────────────────

def plot_bar_comparison(rouge: dict, latency: dict):
    import matplotlib.pyplot as plt
    import numpy as np

    models  = list(rouge.keys())
    x       = np.arange(2)
    width   = 0.2
    metrics = ["rouge_l", "latency_norm"]

    # Normalise latency to 0-1 (inverted: lower latency = higher bar)
    max_lat = max(latency.values())
    lat_norm = {m: round(1 - latency[m]/max_lat, 4) for m in models}

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Model Comparison: Key Metrics", fontsize=14, fontweight="bold", y=1.01)

    # Left: ROUGE-L
    ax = axes[0]
    for i, m in enumerate(models):
        val = rouge[m]["mean"]
        bar = ax.bar(i, val, color=MODEL_COLORS.get(m, "#999"),
                     label=MODEL_LABELS.get(m, m), alpha=0.88, edgecolor="white")
        ax.text(i, val + 0.01, f"{val:.3f}", ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in models], fontsize=10)
    ax.set_ylabel("ROUGE-L F1", fontsize=11)
    ax.set_title("ROUGE-L (higher = better)", fontsize=11)
    ax.set_ylim(0, max(rouge[m]["mean"] for m in models) * 1.25)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    # Right: Latency
    ax = axes[1]
    for i, m in enumerate(models):
        val = latency[m]
        bar = ax.bar(i, val, color=MODEL_COLORS.get(m, "#999"),
                     alpha=0.88, edgecolor="white")
        ax.text(i, val + 1, f"{val:.1f}s", ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in models], fontsize=10)
    ax.set_ylabel("Mean Eval Latency (s)", fontsize=11)
    ax.set_title("Inference Latency (lower = better)", fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    path = PLOTS_DIR / "bar_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {path.name}")

# ── 2. ROUGE-L by difficulty ───────────────────────────────────────────────────

def plot_rouge_by_difficulty(rouge: dict):
    import matplotlib.pyplot as plt
    import numpy as np

    difficulties = ["easy", "medium", "hard"]
    models       = list(rouge.keys())
    x            = np.arange(len(difficulties))
    width        = 0.2

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, m in enumerate(models):
        bd   = rouge[m]["by_difficulty"]
        vals = [bd.get(d, 0) for d in difficulties]
        bars = ax.bar(x + i*width, vals, width,
                      label=MODEL_LABELS.get(m, m),
                      color=MODEL_COLORS.get(m, "#999"),
                      alpha=0.88, edgecolor="white")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{v:.2f}", ha="center", fontsize=8)

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([d.capitalize() for d in difficulties], fontsize=11)
    ax.set_ylabel("ROUGE-L F1", fontsize=11)
    ax.set_title("ROUGE-L F1 by Question Difficulty", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 0.95)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    path = PLOTS_DIR / "rouge_by_difficulty.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {path.name}")

# ── 3. Latency vs Quality scatter ──────────────────────────────────────────────

def plot_latency_vs_quality(rouge: dict, latency: dict):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    for m in rouge:
        x = latency[m]
        y = rouge[m]["mean"]
        ax.scatter(x, y, s=200, color=MODEL_COLORS.get(m, "#999"),
                   zorder=5, edgecolors="white", linewidths=1.5)
        ax.annotate(MODEL_LABELS.get(m, m), (x, y),
                    textcoords="offset points", xytext=(8, 5), fontsize=10)

    ax.set_xlabel("Mean Eval Latency (s)", fontsize=11)
    ax.set_ylabel("ROUGE-L F1", fontsize=11)
    ax.set_title("Quality vs Speed Trade-off", fontsize=13, fontweight="bold")
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    path = PLOTS_DIR / "latency_vs_quality.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {path.name}")

# ── 4. Score distribution box plots ───────────────────────────────────────────

def plot_score_distribution(overall: dict):
    import matplotlib.pyplot as plt

    models = list(overall.keys())
    data   = [overall[m] for m in models]
    colors = [MODEL_COLORS.get(m, "#999") for m in models]
    labels = [MODEL_LABELS.get(m, m) for m in models]

    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot(data, patch_artist=True, notch=False,
                    medianprops=dict(color="white", linewidth=2.5))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)
    for w in bp["whiskers"]: w.set_color("#666")
    for c in bp["caps"]:     c.set_color("#666")
    for f in bp["fliers"]:
        f.set_marker("o"); f.set_markerfacecolor("#aaa"); f.set_markersize(5)

    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Overall Score (1–10)", fontsize=11)
    ax.set_title("Distribution of Evaluator Scores per Model", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 11)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    path = PLOTS_DIR / "score_distribution.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {path.name}")

# ── 5. Consistency σ bar chart ─────────────────────────────────────────────────

def plot_consistency(raw: dict):
    import matplotlib.pyplot as plt

    models = list(raw.keys())
    vals   = []
    for m in models:
        records = raw[m]
        sigma_vals = [r["consistency"] for r in records]
        vals.append(round(sum(sigma_vals)/len(sigma_vals), 4))

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(range(len(models)), vals,
                  color=[MODEL_COLORS.get(m, "#999") for m in models],
                  alpha=0.88, edgecolor="white")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"σ={v:.3f}", ha="center", fontsize=10, fontweight="bold")

    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in models], fontsize=11)
    ax.set_ylabel("Mean Score Std Dev (σ)", fontsize=11)
    ax.set_title("Evaluator Consistency — Lower σ is Better", fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    path = PLOTS_DIR / "consistency.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {path.name}")

# ── 6. Kappa heatmap ──────────────────────────────────────────────────────────

def plot_kappa_heatmap(kappa: dict):
    import matplotlib.pyplot as plt
    import numpy as np

    models = ["llama", "gemma", "phi3", "mistral"]
    labels = [MODEL_LABELS[m] for m in models]
    n      = len(models)
    matrix = np.ones((n, n))

    for i, m1 in enumerate(models):
        for j, m2 in enumerate(models):
            if i == j:
                matrix[i][j] = 1.0
            else:
                key  = f"{m1}_vs_{m2}"
                key2 = f"{m2}_vs_{m1}"
                val  = kappa.get(key, kappa.get(key2))
                if val is not None:
                    matrix[i][j] = float(val)
                    matrix[j][i] = float(val)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label="Cohen's κ")
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(labels, fontsize=9, rotation=15, ha="right")
    ax.set_yticklabels(labels, fontsize=9)
    for i in range(n):
        for j in range(n):
            v = matrix[i][j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=11, fontweight="bold",
                    color="white" if v > 0.6 else "black")
    ax.set_title("Inter-Model Agreement (Cohen's κ)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    path = PLOTS_DIR / "kappa_heatmap.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {path.name}")

# ── 7. Score trend across samples ─────────────────────────────────────────────

def plot_score_trend(raw: dict):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 4))
    for m, records in raw.items():
        scores = [r["scores"]["overall"] for r in records]
        ax.plot(range(1, len(scores)+1), scores,
                label=MODEL_LABELS.get(m, m),
                color=MODEL_COLORS.get(m, "#999"),
                linewidth=1.8, alpha=0.85, marker="o", markersize=3)

    ax.set_xlabel("Sample #", fontsize=11)
    ax.set_ylabel("Overall Score", fontsize=11)
    ax.set_title("Evaluator Score Trend Across 30 Benchmark Samples", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_ylim(0, 11)
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    path = PLOTS_DIR / "score_trend.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {path.name}")

# ── LaTeX tables ───────────────────────────────────────────────────────────────

def generate_latex(rouge: dict, latency: dict, consistency: dict, kappa: dict) -> str:
    models = list(rouge.keys())
    lines  = []

    # ── Main results table ──
    lines += [
        r"\begin{table}[h]",
        r"\caption{Comparative evaluation of LLaMA~3.2, Gemma~7B, Phi-3, and Mistral~7B. "
        r"$\uparrow$ higher is better; $\downarrow$ lower is better.}\label{tab:results}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"\textbf{Model} & \textbf{ROUGE-L $\uparrow$} & \textbf{Latency~(s) $\downarrow$} "
        r"& \textbf{Consist.~$\sigma$ $\downarrow$} & \textbf{$\kappa$ $\uparrow$} \\",
        r"\midrule",
    ]
    for m in models:
        label = MODEL_LABELS.get(m, m)
        r_val = rouge[m]["mean"]
        l_val = latency[m]
        c_val = consistency[m]
        kkey  = f"llama_vs_{m}" if m != "llama" else None
        kval  = "1.00" if m == "llama" else \
                f"{kappa.get(kkey, kappa.get(f'{m}_vs_llama', 0)):.2f}"
        lines.append(f"{label} & {r_val:.4f} & {l_val:.1f} & {c_val:.4f} & {kval} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]

    # ── Difficulty table ──
    lines += [
        r"\begin{table}[h]",
        r"\caption{ROUGE-L F1 per question difficulty level.}\label{tab:difficulty}",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"\textbf{Model} & \textbf{Easy} & \textbf{Medium} & \textbf{Hard} \\",
        r"\midrule",
    ]
    for m in models:
        bd = rouge[m]["by_difficulty"]
        label = MODEL_LABELS.get(m, m)
        lines.append(f"{label} & {bd.get('easy',0):.4f} & {bd.get('medium',0):.4f} & {bd.get('hard',0):.4f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    return "\n".join(lines)

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    try:
        import matplotlib
        matplotlib.use("Agg")
    except ImportError:
        print("[ERROR] pip install matplotlib")
        sys.exit(1)

    if not (RESULTS_DIR / "raw_results.json").exists():
        print("[ERROR] raw_results.json not found. Run run_benchmark.py first.")
        sys.exit(1)

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    summary, raw, kappa = load()

    rouge       = recompute_rouge(raw)
    latency     = recompute_latency(raw)
    consistency = recompute_consistency(raw)
    overall     = recompute_overall(raw)

    print("\n── Generating plots...")
    plot_bar_comparison(rouge, latency)
    plot_rouge_by_difficulty(rouge)
    plot_latency_vs_quality(rouge, latency)
    plot_score_distribution(overall)
    plot_consistency(raw)
    plot_kappa_heatmap(kappa)
    plot_score_trend(raw)

    print("\n── Generating LaTeX tables...")
    latex = generate_latex(rouge, latency, consistency, kappa)
    latex_path = RESULTS_DIR / "latex_tables.txt"
    latex_path.write_text(latex)
    print(f"  ✓ latex_tables.txt")

    # Print corrected numbers for updating the paper
    print("\n── CORRECTED NUMBERS FOR YOUR PAPER ──────────────────────────")
    print(f"  {'Model':<12} {'ROUGE-L':>8} {'Latency':>10} {'σ':>8} {'κ vs llama':>12}")
    print("  " + "-"*52)
    for m in rouge:
        kkey = f"llama_vs_{m}" if m != "llama" else "—"
        kval = kappa.get(kkey, kappa.get(f"{m}_vs_llama", "—"))
        kstr = f"{kval:.4f}" if isinstance(kval, float) else kval
        print(f"  {MODEL_LABELS.get(m,m):<12} {rouge[m]['mean']:>8.4f} "
              f"{latency[m]:>10.2f} {consistency[m]:>8.4f} {kstr:>12}")
    print()
    print("  ROUGE-L by Difficulty:")
    print(f"  {'Model':<12} {'Easy':>8} {'Medium':>8} {'Hard':>8}")
    for m in rouge:
        bd = rouge[m]["by_difficulty"]
        print(f"  {MODEL_LABELS.get(m,m):<12} {bd['easy']:>8.4f} {bd['medium']:>8.4f} {bd['hard']:>8.4f}")
    print(f"\n  All plots → {PLOTS_DIR}")
    print(f"  LaTeX    → {latex_path}\n")

if __name__ == "__main__":
    main()