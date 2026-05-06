"""
MUSIQ Evaluation on Validation Split Only
Uses the SAME train/test split as your fine-tuned model evaluation
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr, kendalltau
from sklearn.model_selection import train_test_split
import os
import sys
import random

# Set the SAME random seed as your evaluation script
SEED = 50
random.seed(SEED)
np.random.seed(SEED)

# Add path to your modules
sys.path.append('/home/debajyoti/paridhi_mtp/NAS/MTP/Persuasive_Image/Persuasive_Image')
from simple_data_preprocess import extract_text_data

# Import baseline for metrics
from musiq_baseline_2 import MUSIQAdaptiveBaseline
baseline = MUSIQAdaptiveBaseline()

# Paths - from your evaluation script
data_root = "/home/debajyoti/paridhi_mtp/NAS/MTP/Persuasive_Image/Persuasive_Image/home/debajyoti/paridhi_mtp/product_images_real"
dataset_image = os.path.join(data_root, "dataset_image_new")
dataset_user_preferred = os.path.join(data_root, "final_data")
MUSIQ_CSV = "./musiq_baseline_results/musiq_scores.csv"
OUT_CSV = "musiq_eval_validation_only.csv"


def load_dataset():
    """
    Load dataset EXACTLY like your evaluation script
    This ensures we get the same groups
    """
    data = []
    MAX_IMAGES = 4

    print("Loading dataset (matching your eval script)...")
    for category in os.listdir(dataset_image):
        category_path = os.path.join(dataset_image, category)
        if os.path.isdir(category_path):
            for group in os.listdir(category_path):
                group_path = os.path.join(dataset_image, category, group)
                if os.path.isdir(group_path):
                    images = sorted([
                        os.path.join(group_path, img)
                        for img in os.listdir(group_path)
                        if img.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
                    ])
                    
                    if len(images) > MAX_IMAGES or len(images) == 0:
                        continue
                    
                    user_preferred_path = os.path.join(dataset_user_preferred, category, group, "user_output.txt")
                    
                    if os.path.exists(user_preferred_path):
                        data.append({
                            "images": images,
                            "category": category,
                            "group": group,
                            "ground_truth_path": user_preferred_path
                        })

    print(f"Loaded {len(data)} groups with ground truth")
    return data


def load_gt_scores(gt_file_path):
    """Load ground truth scores from file"""
    try:
        extracted_info, _ = extract_text_data(gt_file_path)
        
        if not extracted_info:
            return None
        
        extracted_info.sort(key=lambda x: x["image_num"])
        scores = [item.get("score", 0) for item in extracted_info]
        
        if not all(score is not None for score in scores):
            return None
        
        return scores
    
    except Exception as e:
        return None


def main():
    """
    Evaluate MUSIQ on validation split only
    """
    # Load full dataset
    all_data = load_dataset()
    
    # Split EXACTLY like your evaluation script
    # Using random_state=45 (from your script)
    train_data, test_data = train_test_split(all_data, test_size=0.05, random_state=45)
    
    print(f"\n✓ Total dataset: {len(all_data)} groups")
    print(f"✓ Training set: {len(train_data)} groups")
    print(f"✓ Validation set: {len(test_data)} groups")
    print(f"\nEvaluating MUSIQ on validation set only...\n")
    
    # Load MUSIQ scores
    musiq_df = pd.read_csv(MUSIQ_CSV)
    
    rows = []
    processed = 0
    skipped = 0
    
    for sample in test_data:
        category = sample["category"]
        group = sample["group"]
        gt_file = sample["ground_truth_path"]
        
        processed += 1
        
        # Get MUSIQ scores for this group
        musiq_group = musiq_df[
            (musiq_df['category'] == category) & 
            (musiq_df['group'] == group)
        ]
        
        if len(musiq_group) == 0:
            print(f"  Skipped {category}/{group}: No MUSIQ scores found")
            skipped += 1
            continue
        
        musiq_scores = musiq_group.sort_values('image_path')['musiq_score'].tolist()
        
        # Load ground truth
        gt_scores = load_gt_scores(gt_file)
        
        if gt_scores is None:
            print(f"  Skipped {category}/{group}: Could not load ground truth")
            skipped += 1
            continue
        
        # Check length match
        if len(gt_scores) != len(musiq_scores):
            print(f"  Skipped {category}/{group}: Length mismatch (GT={len(gt_scores)}, MUSIQ={len(musiq_scores)})")
            skipped += 1
            continue
        
        # Calculate metrics
        metrics = baseline.calculate_metrics(gt_scores, musiq_scores)
        
        if metrics is None:
            print(f"  Skipped {category}/{group}: Metrics calculation failed")
            skipped += 1
            continue
        
        rows.append({
            "category": category,
            "group": group,
            "num_images": len(gt_scores),
            **metrics
        })
    
    if not rows:
        print("\n❌ ERROR: No valid comparisons found!")
        return
    
    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT_CSV, index=False)
    
    print(f"\n✅ Saved: {OUT_CSV}")
    print(f"   Validation samples evaluated: {len(rows)}/{len(test_data)}")
    print(f"   Skipped: {skipped}")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("MUSIQ BASELINE - VALIDATION SET ONLY")
    print("="*60)
    print(f"\nTotal validation samples: {len(rows)}")
    print(f"\n{'Metric':<30} {'Mean':>10} {'Std':>10}")
    print("-"*52)
    
    metrics_to_show = [
        'mse', 'rank_agreement', 'normalized_ranking_loss', 
        'top_accuracy', 'spearman_correlation', 'kendall_tau', 
        'pearson_correlation'
    ]
    
    for metric in metrics_to_show:
        mean_val = out_df[metric].mean()
        std_val = out_df[metric].std()
        print(f"{metric:<30} {mean_val:>10.4f} {std_val:>10.4f}")
    
    print("="*60)
    
    # Category-wise breakdown
    print("\n" + "="*60)
    print("CATEGORY-WISE RESULTS (Validation Set)")
    print("="*60)
    
    if len(out_df) > 0:
        category_stats = out_df.groupby('category').agg({
            'mse': 'mean',
            'rank_agreement': 'mean',
            'top_accuracy': 'mean',
            'spearman_correlation': 'mean',
            'kendall_tau': 'mean',
            'num_images': 'count'
        }).round(4)
        
        category_stats = category_stats.rename(columns={'num_images': 'num_groups'})
        print(category_stats)
    
    print("\n✅ Evaluation complete!")
    print(f"\nThis matches your fine-tuned model evaluation:")
    print(f"  - Same random seed: {SEED}")
    print(f"  - Same train/test split: test_size=0.05, random_state=45")
    print(f"  - Same validation set: {len(test_data)} groups")


if __name__ == "__main__":
    main()