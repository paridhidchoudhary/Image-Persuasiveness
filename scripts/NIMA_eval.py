"""
NIMA Evaluation — Fixed Version

Evaluates NIMA on the SAME test split as the finetuned model:
  test_size=0.15, random_state=48

Fixes from original:
  1. Correct split: test_size=0.15, random_state=48
  2. Image ordering: sorted alphabetically — matches rest of pipeline
  3. Tied GT groups filtered out
  4. calculate_rank_agreement uses positional formula matching eval_scorehead.py
  5. All metrics computed: kendall_tau, spearman_rho, agreement,
     top_accuracy, norm_ranking_loss (so NIMA can go into all box plots)
  6. Output saved as nima_summary.json matching musiq_summary.json format
     so make_boxplots_with_musiq.py can load both identically

Run nima_score.py first, then this.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.stats import kendalltau, spearmanr
import os
import sys
import random
import json

sys.path.append('/home/debajyoti/paridhi_mtp/nas_final/mtp/Persuasive_Image/Persuasive_Image')
from simple_data_preprocess import extract_text_data

# ── Config — MUST match eval_scorehead.py exactly ─────────────────────────────
SEED         = 50
TEST_SIZE    = 0.15
RANDOM_STATE = 48

data_root              = "home/debajyoti/paridhi_mtp/product_images_real"
dataset_image          = os.path.join(data_root, "dataset_image_new")
dataset_user_preferred = os.path.join(data_root, "final_data")
NIMA_CSV               = "./nima_baseline_results/nima_scores.csv"
output_dir             = "./nima_baseline_results"
os.makedirs(output_dir, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)

# ── Metric helpers — identical to eval_scorehead.py ───────────────────────────
def get_ranking_from_scores(scores, handle_ties=True):
    if not scores:
        return []
    pairs = [(i, s) for i, s in enumerate(scores)]
    sorted_pairs = sorted(pairs, key=lambda x: x[1], reverse=True)
    ranking = [0] * len(scores)
    if handle_ties:
        current_rank = 0
        last_score   = None
        for i, (idx, score) in enumerate(sorted_pairs):
            if i > 0 and score == last_score:
                ranking[idx] = current_rank
            else:
                current_rank  = i
                ranking[idx]  = current_rank
            last_score = score
    else:
        for rank, (idx, _) in enumerate(sorted_pairs):
            ranking[idx] = rank
    return ranking

def calculate_kendall_tau(s1, s2):
    if len(s1) != len(s2) or len(s1) < 2:
        return 0.0
    tau, _ = kendalltau(s1, s2)
    return 0.0 if np.isnan(tau) else float(tau)

def calculate_spearman_rho(s1, s2):
    if len(s1) != len(s2) or len(s1) < 2:
        return 0.0
    rho, _ = spearmanr(s1, s2)
    return 0.0 if np.isnan(rho) else float(rho)

def calculate_rank_agreement(r1, r2):
    # Positional agreement — same formula as eval_scorehead.py
    if len(r1) != len(r2):
        return 0.0
    return sum(a == b for a, b in zip(r1, r2)) / len(r1)

def calculate_normalized_ranking_loss(r1, r2):
    if len(r1) != len(r2):
        return float('inf')
    return float(np.linalg.norm(np.array(r1) - np.array(r2)) / len(r1))

def calculate_top_accuracy(r1, r2):
    if len(r1) != len(r2):
        return 0
    return 1 if np.argmin(r1) == np.argmin(r2) else 0

def calculate_mse(s1, s2):
    if len(s1) != len(s2):
        return float('inf')
    return float(np.mean((np.array(s1) - np.array(s2)) ** 2))

# ── Load dataset — same logic as eval_scorehead.py ────────────────────────────
def load_dataset():
    data = []
    MAX_IMAGES = 4
    for category in os.listdir(dataset_image):
        category_path = os.path.join(dataset_image, category)
        if not os.path.isdir(category_path):
            continue
        for group in os.listdir(category_path):
            group_path = os.path.join(dataset_image, category, group)
            if not os.path.isdir(group_path):
                continue
            images = sorted([
                f for f in os.listdir(group_path)
                if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
            ])
            if len(images) > MAX_IMAGES or len(images) == 0:
                continue
            gt_path = os.path.join(dataset_user_preferred, category, group, "user_output.txt")
            if os.path.exists(gt_path):
                data.append({
                    "images":   images,   # sorted basenames
                    "category": category,
                    "group":    group,
                    "gt_path":  gt_path,
                })
    return data

def extract_gt_scores(gt_path):
    try:
        extracted_info, _ = extract_text_data(gt_path)
        if not extracted_info:
            return None
        extracted_info.sort(key=lambda x: x["image_num"])
        scores = [item.get("score", 0) for item in extracted_info]
        if all(s is not None for s in scores):
            return scores
    except:
        pass
    return None

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("Loading dataset...")
    all_data = load_dataset()
    print(f"Total groups: {len(all_data)}")

    _, test_data = train_test_split(
        all_data, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"Test set: {len(test_data)} groups "
          f"(test_size={TEST_SIZE}, random_state={RANDOM_STATE})")

    nima_df = pd.read_csv(NIMA_CSV)
    print(f"Loaded NIMA scores: {len(nima_df)} rows")

    results = []
    skipped = 0

    for sample in test_data:
        category = sample["category"]
        group    = sample["group"]
        images   = sample["images"]   # sorted basenames
        gt_path  = sample["gt_path"]

        # Ground truth
        gt_scores = extract_gt_scores(gt_path)
        if not gt_scores:
            skipped += 1
            continue

        # Skip all-tied GT — same as finetuned eval
        if len(set(gt_scores)) == 1:
            skipped += 1
            continue

        # Get NIMA scores for this group
        nima_group = nima_df[
            (nima_df['category'] == category) &
            (nima_df['group']    == group)
        ].copy()

        if len(nima_group) == 0:
            print(f"  Skipping {category}/{group}: no NIMA scores found in CSV")
            skipped += 1
            continue

        # Sort by image_name alphabetically — matches sorted() used everywhere
        nima_group  = nima_group.sort_values('image_name')
        nima_scores = nima_group['nima_score'].tolist()
        nima_names  = nima_group['image_name'].tolist()

        # Verify count matches
        if len(nima_scores) != len(gt_scores):
            print(f"  Skipping {category}/{group}: "
                  f"NIMA has {len(nima_scores)} images, GT has {len(gt_scores)}")
            skipped += 1
            continue

        # Warn if image names don't match (but still proceed — order is consistent)
        if nima_names != images:
            print(f"  Warning {category}/{group}: image name mismatch")
            print(f"    NIMA:    {nima_names}")
            print(f"    Dataset: {images}")

        # Compute all metrics
        gt_ranking   = get_ranking_from_scores(gt_scores,   handle_ties=True)
        nima_ranking = get_ranking_from_scores(nima_scores, handle_ties=True)

        results.append({
            "category":          category,
            "group":             group,
            "n_images":          len(gt_scores),
            "gt_scores":         gt_scores,
            "nima_scores":       nima_scores,
            "kendall_tau":       calculate_kendall_tau(gt_scores, nima_scores),
            "spearman_rho":      calculate_spearman_rho(gt_scores, nima_scores),
            "agreement":         calculate_rank_agreement(gt_ranking, nima_ranking),
            "top_accuracy":      calculate_top_accuracy(gt_ranking, nima_ranking),
            "norm_ranking_loss": calculate_normalized_ranking_loss(gt_ranking, nima_ranking),
            "mse":               calculate_mse(gt_scores, nima_scores),
        })

    print(f"\nEvaluated: {len(results)}  |  Skipped: {skipped}")

    if not results:
        print("ERROR: No valid results.")
        return

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, "nima_eval_results.csv"), index=False)

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n===== NIMA EVALUATION SUMMARY =====")
    print(f"Total test samples: {len(test_data)}  |  Valid: {len(results)}")

    for metric, higher_better in [
        ("top_accuracy",      True),
        ("agreement",         True),
        ("kendall_tau",       True),
        ("spearman_rho",      True),
        ("norm_ranking_loss", False),
        ("mse",               False),
    ]:
        arrow = "↑" if higher_better else "↓"
        print(f"  {metric:25s} ({arrow}): {df[metric].mean():.4f}")

    print("\n===== CATEGORY-WISE =====")
    cat_stats = df.groupby("category").agg(
        top_accuracy=("top_accuracy",  "mean"),
        kendall_tau =("kendall_tau",   "mean"),
        spearman_rho=("spearman_rho",  "mean"),
        agreement   =("agreement",     "mean"),
        n_groups    =("group",         "count"),
    ).round(3)
    print(cat_stats.to_string())

    # ── Save summary JSON — same format as musiq_summary.json ─────────────────
    # This allows make_boxplots_with_musiq.py to load NIMA identically
    summary = {
        "nima": {
            "top_accuracy":      float(df["top_accuracy"].mean()),
            "agreement":         float(df["agreement"].mean()),
            "kendall_tau":       float(df["kendall_tau"].mean()),
            "spearman_rho":      float(df["spearman_rho"].mean()),
            "norm_ranking_loss": float(df["norm_ranking_loss"].mean()),
            "n":                 len(results),
            "per_sample": {
                row["category"] + "/" + row["group"]: {
                    "kendall_tau":       row["kendall_tau"],
                    "spearman_rho":      row["spearman_rho"],
                    "agreement":         row["agreement"],
                    "top_accuracy":      row["top_accuracy"],
                    "norm_ranking_loss": row["norm_ranking_loss"],
                }
                for row in results
            }
        }
    }

    with open(os.path.join(output_dir, "nima_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {output_dir}/")
    print("Run make_boxplots_with_musiq_nima.py to add NIMA to box plots.")

if __name__ == "__main__":
    main()