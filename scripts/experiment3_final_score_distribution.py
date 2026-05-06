"""
Experiment 3: Score Distribution Analysis

Loads detailed_stats.json and produces:
1. Distribution of predicted scores vs GT scores (scatter + histogram)
2. Per-group score spread analysis (does model discriminate well?)
3. Score calibration plot — how predicted ranking matches GT ranking

Run after eval_scorehead.py has produced detailed_stats.json.
No GPU needed — works entirely from saved results.
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

STATS_FILE = "./listwise_model_evaluation_scorehead_only_lora_ranking_with_explanations/detailed_stats.json"
OUTPUT_DIR = "./experiment_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(STATS_FILE) as f:
    data = json.load(f)
results = data["detailed_results"]
print(f"Loaded {len(results)} test samples")

# ── Extract scores ─────────────────────────────────────────────────────────────
all_gt   = []   # raw GT scores (1-100)
all_pred = []   # predicted scores (0-1)
all_gt_norm = []  # GT scores normalised to 0-1 for fair comparison

# Per-group score ranges — does the model spread scores out enough?
group_pred_ranges = []
group_gt_ranges   = []
group_sizes       = []

for r in results:
    gt   = r["ground_truth_scores"]
    pred = r["finetuned_scores"]
    if not gt or not pred or len(gt) != len(pred):
        continue

    all_gt.extend(gt)
    all_pred.extend(pred)

    gt_min, gt_max = min(gt), max(gt)
    gt_norm = [(s - gt_min) / (gt_max - gt_min + 1e-8) for s in gt]
    all_gt_norm.extend(gt_norm)

    group_pred_ranges.append(max(pred) - min(pred))
    group_gt_ranges.append(max(gt) - min(gt))
    group_sizes.append(len(gt))

all_gt      = np.array(all_gt)
all_pred    = np.array(all_pred)
all_gt_norm = np.array(all_gt_norm)

print(f"\nPredicted score stats: min={all_pred.min():.3f}  max={all_pred.max():.3f}"
      f"  mean={all_pred.mean():.3f}  std={all_pred.std():.3f}")
print(f"GT score stats:        min={all_gt.min():.1f}  max={all_gt.max():.1f}"
      f"  mean={all_gt.mean():.1f}  std={all_gt.std():.1f}")

pearson_r, pearson_p = pearsonr(all_gt_norm, all_pred)
print(f"\nPearson r (pred vs normalised GT): {pearson_r:.4f}  (p={pearson_p:.4f})")

# ── Figure 1: Score distributions ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle("Score Distribution Analysis", fontsize=13, fontweight="bold")

# Panel 1: histogram of predicted scores
axes[0].hist(all_pred, bins=30, color="#2196F3", alpha=0.7, edgecolor="white")
axes[0].set_title("Predicted Score Distribution\n(score head output)", fontweight="bold")
axes[0].set_xlabel("Predicted Score (0–1)")
axes[0].set_ylabel("Count")
axes[0].axvline(all_pred.mean(), color="black", linestyle="--",
                label=f"Mean = {all_pred.mean():.3f}")
axes[0].legend(fontsize=9)
axes[0].grid(axis="y", alpha=0.3, linestyle=":")

# Panel 2: scatter — predicted vs normalised GT
axes[1].scatter(all_gt_norm, all_pred, alpha=0.3, s=20, color="#9C27B0", edgecolors="none")
axes[1].plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Perfect calibration")
axes[1].set_title(f"Pred vs Normalised GT\n(Pearson r = {pearson_r:.3f})", fontweight="bold")
axes[1].set_xlabel("Normalised GT Score (0–1)")
axes[1].set_ylabel("Predicted Score (0–1)")
axes[1].legend(fontsize=9)
axes[1].grid(alpha=0.3, linestyle=":")

# Panel 3: per-group score range comparison
axes[2].scatter(group_gt_ranges, group_pred_ranges, alpha=0.5, s=30,
                color="#FF9800", edgecolors="white", linewidths=0.5)
axes[2].set_title("Per-Group Score Spread\n(GT range vs Predicted range)", fontweight="bold")
axes[2].set_xlabel("GT Score Range (points)")
axes[2].set_ylabel("Predicted Score Range (0–1 scale)")
axes[2].grid(alpha=0.3, linestyle=":")

# Colour by group size
scatter = axes[2].scatter(group_gt_ranges, group_pred_ranges,
                          c=group_sizes, cmap="viridis", alpha=0.6, s=35,
                          edgecolors="white", linewidths=0.3)
plt.colorbar(scatter, ax=axes[2], label="Group size (# images)")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "score_distributions.png"), dpi=150, bbox_inches="tight")
print(f"Saved score_distributions.png")
plt.show()

# ── Figure 2: Ranking calibration ─────────────────────────────────────────────
# For each pair within a group where GT says i > j,
# what fraction does the model also predict i > j?
concordant = 0
discordant = 0
tied       = 0

for r in results:
    gt   = r["ground_truth_scores"]
    pred = r["finetuned_scores"]
    if not gt or not pred or len(gt) != len(pred):
        continue
    n = len(gt)
    for i in range(n):
        for j in range(i+1, n):
            if gt[i] == gt[j]:
                continue
            gt_pref   = gt[i] > gt[j]
            pred_pref = pred[i] > pred[j]
            if pred[i] == pred[j]:
                tied += 1
            elif gt_pref == pred_pref:
                concordant += 1
            else:
                discordant += 1

total_pairs = concordant + discordant + tied
print(f"\nPair-level analysis across all groups:")
print(f"  Total comparable pairs: {total_pairs}")
print(f"  Concordant: {concordant} ({100*concordant/total_pairs:.1f}%)")
print(f"  Discordant: {discordant} ({100*discordant/total_pairs:.1f}%)")
print(f"  Tied predictions: {tied} ({100*tied/total_pairs:.1f}%)")

fig, ax = plt.subplots(figsize=(6, 5))
bars = ax.bar(["Concordant\n(correct)", "Discordant\n(wrong)", "Tied\n(no preference)"],
              [concordant, discordant, tied],
              color=["#4CAF50", "#F44336", "#9E9E9E"], alpha=0.8, edgecolor="white")
for bar, val in zip(bars, [concordant, discordant, tied]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f"{100*val/total_pairs:.1f}%", ha="center", fontsize=10, fontweight="bold")
ax.set_title("Pairwise Ranking Accuracy\n(across all image pairs in test set)",
             fontweight="bold")
ax.set_ylabel("Number of pairs")
ax.grid(axis="y", alpha=0.3, linestyle=":")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "pairwise_concordance.png"), dpi=150, bbox_inches="tight")
print(f"Saved pairwise_concordance.png")
plt.show()

# ── Save summary ───────────────────────────────────────────────────────────────
summary = {
    "pred_score_stats": {
        "min": float(all_pred.min()), "max": float(all_pred.max()),
        "mean": float(all_pred.mean()), "std": float(all_pred.std()),
    },
    "pearson_r_pred_vs_gt_norm": float(pearson_r),
    "pearson_p": float(pearson_p),
    "pairwise": {
        "total": total_pairs, "concordant": concordant,
        "discordant": discordant, "tied": tied,
        "concordance_rate": float(concordant / total_pairs),
    }
}
with open(os.path.join(OUTPUT_DIR, "score_distribution_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nAll saved to {OUTPUT_DIR}/")