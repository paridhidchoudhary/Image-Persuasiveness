"""
Box plot — Fine-tuned + 4 VLM baselines + MUSIQ + NIMA

Input : detailed_stats.json  (from eval_scorehead.py)
        musiq_summary.json   (from musiq_eval.py)
        nima_summary.json    (from nima_eval.py)
Output: boxplots_with_musiq_nima.png
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt

# ── Config ─────────────────────────────────────────────────────────────────────
STATS_FILE  = "./listwise_model_evaluation_scorehead_only_lora_ranking_with_explanations/detailed_stats.json"
MUSIQ_FILE  = "/home/debajyoti/paridhi_mtp/nas_final/MTP/Persuasive_Image/Persuasive_Image/MUSIQ/musiq_baseline_results_2/musiq_summary.json"
NIMA_FILE   = "./nima_baseline_results/nima_summary.json"
OUTPUT_DIR  = "./listwise_model_evaluation_scorehead_only_lora_ranking_with_explanations"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "boxplots_with_musiq_nima.png")

BASELINES = ["qwen_zeroshot", "qwen_fewshot", "pixtral_zeroshot", "pixtral_fewshot"]

ALL_MODEL_KEYS = ["finetuned"] + BASELINES + ["musiq", "nima"]
DISPLAY_NAMES  = [
    "Fine-tuned\n(Ours)",
    "Qwen\nZero-shot",
    "Qwen\nFew-shot",
    "Pixtral\nZero-shot",
    "Pixtral\nFew-shot",
    "MUSIQ",
    "NIMA",
]
COLORS = [
    "#2196F3",   # blue       — fine-tuned
    "#FF9800",   # orange     — qwen zeroshot
    "#4CAF50",   # green      — qwen fewshot
    "#9C27B0",   # purple     — pixtral zeroshot
    "#F44336",   # red        — pixtral fewshot
    "#795548",   # brown      — MUSIQ
    "#607D8B",   # blue-grey  — NIMA
]

METRICS = [
    ("kendall_tau",  "Kendall's Tau"),
    ("spearman_rho", "Spearman's Rho"),
    ("agreement",    "Agreement"),
    ("top_accuracy", "Top-1 Accuracy"),
]

# ── Load data ──────────────────────────────────────────────────────────────────
with open(STATS_FILE) as f:
    data = json.load(f)
results = data["detailed_results"]
print(f"Loaded {len(results)} test samples from detailed_stats.json")

with open(MUSIQ_FILE) as f:
    musiq_data = json.load(f)
musiq_per_sample = musiq_data["musiq"]["per_sample"]
print(f"Loaded {len(musiq_per_sample)} MUSIQ per-sample results")

with open(NIMA_FILE) as f:
    nima_data = json.load(f)
nima_per_sample = nima_data["nima"]["per_sample"]
print(f"Loaded {len(nima_per_sample)} NIMA per-sample results")

# ── Extract values ─────────────────────────────────────────────────────────────
def get_vals(results, model_name, metric,
             musiq_per_sample=None, nima_per_sample=None):
    if model_name == "musiq":
        return [v[metric] for v in musiq_per_sample.values() if metric in v]
    if model_name == "nima":
        return [v[metric] for v in nima_per_sample.values() if metric in v]

    vals = []
    for r in results:
        ft = r["metrics"]["finetuned_vs_ground_truth"]
        mm = r["metrics"]["model_metrics"]
        if model_name == "finetuned":
            if metric in ft:
                vals.append(ft[metric])
        else:
            if model_name in mm and metric in mm[model_name]["vs_ground_truth"]:
                vals.append(mm[model_name]["vs_ground_truth"][metric])
    return vals

# Print sample sizes
print("\nSample sizes per model:")
for m, name in zip(ALL_MODEL_KEYS, DISPLAY_NAMES):
    n = len(get_vals(results, m, "top_accuracy", musiq_per_sample, nima_per_sample))
    print(f"  {name.replace(chr(10), ' '):25s}: {n}")

# ── Plot ───────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(20, 11))
fig.suptitle(
    "Model Comparison: Distribution of Ranking Metrics vs Ground Truth\n"
    "(each dot = one test sample)",
    fontsize=13, fontweight="bold", y=1.01
)

np.random.seed(42)  # reproducible jitter

for ax, (metric, title) in zip(axes.flat, METRICS):
    all_data = [
        get_vals(results, m, metric, musiq_per_sample, nima_per_sample)
        for m in ALL_MODEL_KEYS
    ]

    bp = ax.boxplot(
        all_data,
        patch_artist=True,
        notch=False,
        medianprops=dict(color="black", linewidth=2.5),
        whiskerprops=dict(linewidth=1.5),
        capprops=dict(linewidth=1.5),
        flierprops=dict(marker="o", markersize=5, alpha=0.4),
    )

    for patch, color in zip(bp["boxes"], COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)

    for i, (vals, color) in enumerate(zip(all_data, COLORS), 1):
        if not vals:
            continue
        jitter = np.random.normal(0, 0.07, size=len(vals))
        ax.scatter(i + jitter, vals, color=color, alpha=0.55, s=26, zorder=3,
                   edgecolors="white", linewidths=0.5)

    # Vertical separator between VLM baselines and IQA metrics
    ax.axvline(x=5.5, color="gray", linewidth=0.8, linestyle=":", alpha=0.5)

    ax.set_title(title, fontweight="bold", fontsize=11)
    ax.set_xticks(range(1, len(ALL_MODEL_KEYS) + 1))
    ax.set_xticklabels(DISPLAY_NAMES, fontsize=7.5)
    ax.set_ylim(-1.15, 1.32)
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.4)
    ax.grid(axis="y", alpha=0.25, linestyle=":")
    ax.set_ylabel("Score", fontsize=10)

    for i, (vals, color) in enumerate(zip(all_data, COLORS), 1):
        if not vals:
            continue
        ax.text(i, 1.22, f"μ={np.mean(vals):.2f}",
                ha="center", fontsize=7, fontweight="bold", color=color)
        ax.text(i, -1.12, f"n={len(vals)}",
                ha="center", fontsize=7, color="gray")

    # Label the two sections
    ax.text(3.0,  1.30, "VLM Models",  ha="center", fontsize=8,
            color="gray", style="italic")
    ax.text(6.5,  1.30, "IQA Metrics", ha="center", fontsize=8,
            color="gray", style="italic")

plt.tight_layout()
os.makedirs(OUTPUT_DIR, exist_ok=True)
plt.savefig(OUTPUT_FILE, dpi=150, bbox_inches="tight")
print(f"\nSaved to {OUTPUT_FILE}")
plt.show()