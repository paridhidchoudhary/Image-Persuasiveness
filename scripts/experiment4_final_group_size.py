"""
Experiment 4: Performance by Group Size

Breaks down all metrics by number of images in the group (n=2, n=3, n=4).
Also includes all baseline models for comparison.

Answers: Does the model perform better on simpler 2-image groups?
         How does performance degrade as group size increases?
         Does this pattern hold for baselines too?

Works entirely from detailed_stats.json — no GPU needed.
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

STATS_FILE = "./listwise_model_evaluation_scorehead_only_lora_ranking_with_explanations/detailed_stats.json"
OUTPUT_DIR = "./experiment_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(STATS_FILE) as f:
    data = json.load(f)
results = data["detailed_results"]
print(f"Loaded {len(results)} test samples")

BASELINES = ["qwen_zeroshot", "qwen_fewshot", "pixtral_zeroshot", "pixtral_fewshot"]
METRICS   = ["kendall_tau", "spearman_rho", "agreement", "top_accuracy"]

# ── Collect per-group-size metrics ─────────────────────────────────────────────
# Structure: size -> model -> metric -> [values]
by_size = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

for r in results:
    gt   = r["ground_truth_scores"]
    pred = r["finetuned_scores"]
    if not gt or not pred: continue
    n = len(gt)
    if n not in (2, 3, 4): continue

    ft = r["metrics"]["finetuned_vs_ground_truth"]
    mm = r["metrics"]["model_metrics"]

    for metric in METRICS:
        if metric in ft:
            by_size[n]["finetuned"][metric].append(ft[metric])

    for bl in BASELINES:
        if bl in mm:
            bl_gt = mm[bl]["vs_ground_truth"]
            for metric in METRICS:
                if metric in bl_gt:
                    by_size[n][bl][metric].append(bl_gt[metric])

# ── Print summary table ────────────────────────────────────────────────────────
print("\n===== PERFORMANCE BY GROUP SIZE =====")
all_models = ["finetuned"] + BASELINES
model_labels = {
    "finetuned":        "Fine-tuned (Ours)",
    "qwen_zeroshot":    "Qwen Zero-shot",
    "qwen_fewshot":     "Qwen Few-shot",
    "pixtral_zeroshot": "Pixtral Zero-shot",
    "pixtral_fewshot":  "Pixtral Few-shot",
}

for size in [2, 3, 4]:
    n_samples = len(by_size[size]["finetuned"]["kendall_tau"])
    print(f"\n── Group size n={size}  ({n_samples} test samples) ──")
    print(f"  {'Model':<22} {'Kendall':>8} {'Spearman':>9} {'Agreement':>10} {'Top Acc':>8}")
    print(f"  {'─'*22} {'─'*8} {'─'*9} {'─'*10} {'─'*8}")
    for model in all_models:
        vals = by_size[size][model]
        if not vals.get("kendall_tau"): continue
        k = np.mean(vals["kendall_tau"])
        s = np.mean(vals["spearman_rho"])
        a = np.mean(vals["agreement"])
        t = np.mean(vals["top_accuracy"])
        label = model_labels.get(model, model)
        marker = " ◀" if model == "finetuned" else ""
        print(f"  {label:<22} {k:>8.4f} {s:>9.4f} {a:>10.4f} {t:>8.4f}{marker}")

# ── Figure 1: Kendall's Tau by group size for all models ──────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Performance by Number of Images in Group", fontsize=13, fontweight="bold")

colors = {
    "finetuned":        "#2196F3",
    "qwen_zeroshot":    "#FF9800",
    "qwen_fewshot":     "#4CAF50",
    "pixtral_zeroshot": "#9C27B0",
    "pixtral_fewshot":  "#F44336",
}
linestyles = {
    "finetuned":        "-",
    "qwen_zeroshot":    "--",
    "qwen_fewshot":     "--",
    "pixtral_zeroshot": "-.",
    "pixtral_fewshot":  "-.",
}

sizes = [2, 3, 4]

for ax, metric, title in zip(axes,
                              ["kendall_tau", "top_accuracy"],
                              ["Kendall's Tau (↑)", "Top-1 Accuracy (↑)"]):
    for model in all_models:
        means = []
        stds  = []
        valid_sizes = []
        for s in sizes:
            vals = by_size[s][model].get(metric, [])
            if vals:
                means.append(np.mean(vals))
                stds.append(np.std(vals))
                valid_sizes.append(s)

        if not means: continue

        label = model_labels.get(model, model)
        lw    = 2.5 if model == "finetuned" else 1.5
        ax.plot(valid_sizes, means,
                color=colors[model], linestyle=linestyles[model],
                linewidth=lw, marker="o", markersize=7, label=label)
        ax.fill_between(valid_sizes,
                        [m - s for m, s in zip(means, stds)],
                        [m + s for m, s in zip(means, stds)],
                        alpha=0.1, color=colors[model])

    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Number of Images in Group")
    ax.set_ylabel("Score")
    ax.set_xticks(sizes)
    ax.set_xticklabels([f"n={s}" for s in sizes])
    ax.set_ylim(0, 1.1)
    ax.grid(alpha=0.3, linestyle=":")
    ax.legend(fontsize=8, loc="lower left")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "performance_by_group_size.png"),
            dpi=150, bbox_inches="tight")
print(f"\nSaved performance_by_group_size.png")
plt.show()

# ── Figure 2: Sample counts per group size ────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 4))
counts = [len(by_size[s]["finetuned"]["kendall_tau"]) for s in sizes]
bars   = ax.bar([f"n={s}" for s in sizes], counts,
                color=["#2196F3", "#4CAF50", "#FF9800"], alpha=0.8, edgecolor="white")
for bar, c in zip(bars, counts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            str(c), ha="center", fontsize=11, fontweight="bold")
ax.set_title("Test Sample Distribution by Group Size", fontweight="bold")
ax.set_ylabel("Number of test groups")
ax.grid(axis="y", alpha=0.3, linestyle=":")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "group_size_distribution.png"),
            dpi=150, bbox_inches="tight")
print(f"Saved group_size_distribution.png")
plt.show()

# ── Save summary ───────────────────────────────────────────────────────────────
summary = {}
for size in sizes:
    summary[f"n{size}"] = {}
    for model in all_models:
        vals = by_size[size][model]
        if not vals: continue
        summary[f"n{size}"][model] = {
            m: {"mean": float(np.mean(v)), "std": float(np.std(v)), "n": len(v)}
            for m, v in vals.items() if v
        }

with open(os.path.join(OUTPUT_DIR, "group_size_analysis.json"), "w") as f:
    json.dump(summary, f, indent=2)
print(f"All saved to {OUTPUT_DIR}/")