"""
benchmark/visualize.py — Generates all plots and tables for the research paper.

Run AFTER run_benchmark.py has completed:
    python benchmark/visualize.py

Outputs (saved to benchmark/results/plots/):
    1. bar_comparison.png       — main metric comparison bar chart
    2. bertscore_by_diff.png    — BERTScore per difficulty grouped bar
    3. latency_vs_quality.png   — scatter: latency vs BERTScore
    4. score_distribution.png   — box plots of overall scores per model
    5. kappa_heatmap.png        — inter-model agreement heatmap
    6. latex_table.txt          — ready-to-paste LaTeX table for your paper
"""

import json
from pathlib import Path

RESULTS_DIR = Path("benchmark/results")
PLOTS_DIR   = RESULTS_DIR / "plots"

MODEL_LABELS = {
    "llama": "LLaMA 3.2",
    "gemma": "Gemma 7B",
    "phi3":  "Phi-3",
}
MODEL_COLORS = {
    "llama": "#4f8ef7",
    "gemma": "#2dd4a0",
    "phi3":  "#f7b94f",
}


def load_results() -> tuple[dict, dict, dict]:
    summary = json.loads((RESULTS_DIR / "summary.json").read_text())
    raw     = json.loads((RESULTS_DIR / "raw_results.json").read_text())
    kappa   = json.loads((RESULTS_DIR / "kappa.json").read_text())
    return summary, raw, kappa


# ── 1. Bar chart: main metrics comparison ─────────────────────────────────────

def plot_bar_comparison(summary: dict, ax=None):
    import matplotlib.pyplot as plt
    import numpy as np

    metrics     = ["rouge_l", "bertscore"]
    metric_labels = ["ROUGE-L F1", "BERTScore F1"]
    models      = list(summary.keys())
    x           = np.arange(len(metrics))
    width       = 0.25

    fig, ax = plt.subplots(figsize=(8, 5)) if ax is None else (None, ax)
    for i, model_id in enumerate(models):
        vals = [summary[model_id][m]["mean"] for m in metrics]
        bars = ax.bar(x + i * width, vals, width,
                      label=MODEL_LABELS.get(model_id, model_id),
                      color=MODEL_COLORS.get(model_id, "#999"),
                      alpha=0.88, edgecolor="white", linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_xticks(x + width)
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Model Comparison: Text Quality Metrics", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    if fig:
        path = PLOTS_DIR / "bar_comparison.png"
        fig.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {path}")
        plt.close(fig)


# ── 2. BERTScore by difficulty ─────────────────────────────────────────────────

def plot_bertscore_by_difficulty(summary: dict):
    import matplotlib.pyplot as plt
    import numpy as np

    difficulties  = ["easy", "medium", "hard"]
    models        = list(summary.keys())
    x             = np.arange(len(difficulties))
    width         = 0.25

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, model_id in enumerate(models):
        bd   = summary[model_id].get("bertscore_by_difficulty", {})
        vals = [bd.get(d, 0) for d in difficulties]
        bars = ax.bar(x + i * width, vals, width,
                      label=MODEL_LABELS.get(model_id, model_id),
                      color=MODEL_COLORS.get(model_id, "#999"),
                      alpha=0.88, edgecolor="white", linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x + width)
    ax.set_xticklabels([d.capitalize() for d in difficulties], fontsize=11)
    ax.set_ylabel("BERTScore F1", fontsize=11)
    ax.set_title("BERTScore F1 by Question Difficulty", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    path = PLOTS_DIR / "bertscore_by_diff.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.close(fig)


# ── 3. Latency vs Quality scatter ──────────────────────────────────────────────

def plot_latency_vs_quality(summary: dict):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    for model_id, m in summary.items():
        lat  = m["latency_s"]["mean"]
        bert = m["bertscore"]["mean"]
        ax.scatter(lat, bert, s=180,
                   color=MODEL_COLORS.get(model_id, "#999"),
                   zorder=5, edgecolors="white", linewidths=1.5)
        ax.annotate(MODEL_LABELS.get(model_id, model_id),
                    (lat, bert), textcoords="offset points",
                    xytext=(8, 4), fontsize=10)

    ax.set_xlabel("Mean Inference Latency (s)", fontsize=11)
    ax.set_ylabel("BERTScore F1", fontsize=11)
    ax.set_title("Quality vs Speed Trade-off", fontsize=13, fontweight="bold")
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    path = PLOTS_DIR / "latency_vs_quality.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.close(fig)


# ── 4. Score distribution box plots ───────────────────────────────────────────

def plot_score_distribution(raw: dict):
    import matplotlib.pyplot as plt

    models = list(raw.keys())
    data   = [[r["scores"]["overall"] for r in raw[m]] for m in models]
    colors = [MODEL_COLORS.get(m, "#999") for m in models]
    labels = [MODEL_LABELS.get(m, m) for m in models]

    fig, ax = plt.subplots(figsize=(7, 5))
    bp = ax.boxplot(data, patch_artist=True, notch=False,
                    medianprops=dict(color="white", linewidth=2))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    for whisker in bp["whiskers"]:
        whisker.set_color("#666")
    for cap in bp["caps"]:
        cap.set_color("#666")
    for flier in bp["fliers"]:
        flier.set_marker("o")
        flier.set_markerfacecolor("#999")
        flier.set_markersize(5)

    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Overall Score (1–10)", fontsize=11)
    ax.set_title("Distribution of Overall Scores per Model", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 11)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    path = PLOTS_DIR / "score_distribution.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.close(fig)


# ── 5. Kappa heatmap ──────────────────────────────────────────────────────────

def plot_kappa_heatmap(kappa: dict):
    import matplotlib.pyplot as plt
    import numpy as np

    models = ["llama", "gemma", "phi3"]
    labels = [MODEL_LABELS[m] for m in models]
    n = len(models)
    matrix = np.ones((n, n))

    for i, m1 in enumerate(models):
        for j, m2 in enumerate(models):
            if i == j:
                matrix[i][j] = 1.0
            else:
                key1 = f"{m1}_vs_{m2}"
                key2 = f"{m2}_vs_{m1}"
                val = kappa.get(key1, kappa.get(key2, None))
                if val is not None:
                    matrix[i][j] = val
                    matrix[j][i] = val

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label="Cohen's κ")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, fontsize=10, rotation=15)
    ax.set_yticklabels(labels, fontsize=10)

    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{matrix[i][j]:.2f}",
                    ha="center", va="center", fontsize=11,
                    color="white" if matrix[i][j] > 0.6 else "black",
                    fontweight="bold")

    ax.set_title("Inter-Model Agreement (Cohen's κ)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = PLOTS_DIR / "kappa_heatmap.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.close(fig)


# ── 6. LaTeX table generator ───────────────────────────────────────────────────

def generate_latex_table(summary: dict, kappa: dict) -> str:
    lines = []
    lines.append(r"\begin{table}[h]")
    lines.append(r"\caption{Comparative performance of LLaMA~3.2, Gemma~7B, and Phi-3 across evaluation metrics. "
                 r"$\uparrow$ higher is better; $\downarrow$ lower is better.}\label{tab:results}")
    lines.append(r"\begin{tabular}{lrrrrr}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Model} & \textbf{ROUGE-L $\uparrow$} & \textbf{BERTScore $\uparrow$} "
                 r"& \textbf{$\sigma$ $\downarrow$} & \textbf{$\kappa$ $\uparrow$} "
                 r"& \textbf{Latency (s) $\downarrow$} \\")
    lines.append(r"\midrule")

    for model_id, m in summary.items():
        label = MODEL_LABELS.get(model_id, model_id)
        rouge  = m["rouge_l"]["mean"]
        bert   = m["bertscore"]["mean"]
        sigma  = m["consistency_std"]["mean"]
        lat    = m["latency_s"]["mean"]
        kappa_key = f"llama_vs_{model_id}" if model_id != "llama" else None
        kappa_val = "—" if kappa_key is None else f"{kappa.get(kappa_key, kappa.get(f'{model_id}_vs_llama', 0)):.2f}"
        lines.append(
            f"{label} & {rouge:.2f} & {bert:.2f} & {sigma:.2f} & {kappa_val} & {lat:.1f} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    # Difficulty table
    lines.append("")
    lines.append(r"\begin{table}[h]")
    lines.append(r"\caption{BERTScore F1 per difficulty level.}\label{tab:difficulty}")
    lines.append(r"\begin{tabular}{lrrr}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Model} & \textbf{Easy} & \textbf{Medium} & \textbf{Hard} \\")
    lines.append(r"\midrule")
    for model_id, m in summary.items():
        label = MODEL_LABELS.get(model_id, model_id)
        bd = m.get("bertscore_by_difficulty", {})
        lines.append(
            f"{label} & {bd.get('easy',0):.2f} & {bd.get('medium',0):.2f} & {bd.get('hard',0):.2f} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    try:
        import matplotlib
        matplotlib.use("Agg")   # headless — no display needed
    except ImportError:
        print("[ERROR] matplotlib not installed. Run: pip install matplotlib")
        return

    if not (RESULTS_DIR / "summary.json").exists():
        print("[ERROR] summary.json not found. Run run_benchmark.py first.")
        return

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    summary, raw, kappa = load_results()

    print("\n── Generating plots...")
    plot_bar_comparison(summary)
    plot_bertscore_by_difficulty(summary)
    plot_latency_vs_quality(summary)
    plot_score_distribution(raw)
    plot_kappa_heatmap(kappa)

    print("\n── Generating LaTeX tables...")
    latex = generate_latex_table(summary, kappa)
    latex_path = RESULTS_DIR / "latex_table.txt"
    latex_path.write_text(latex)
    print(f"  Saved: {latex_path}")

    print(f"\n✅ All outputs saved to {PLOTS_DIR}")
    print("   Include these in your paper:")
    print("   - bar_comparison.png      → Section 5, Figure 2")
    print("   - bertscore_by_diff.png   → Section 5, Figure 3")
    print("   - latency_vs_quality.png  → Section 5, Figure 4")
    print("   - score_distribution.png  → Section 5, Figure 5")
    print("   - kappa_heatmap.png       → Section 5, Figure 6")
    print("   - latex_table.txt         → Copy into paper.tex\n")


if __name__ == "__main__":
    main()