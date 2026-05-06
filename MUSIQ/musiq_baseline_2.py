"""
MUSIQ Baseline with Adaptive Score Normalization
This version automatically determines the score range from your data
"""

import torch
import pyiqa
from PIL import Image
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from scipy.stats import spearmanr, kendalltau
import os

class MUSIQAdaptiveBaseline:
    def __init__(self, device=None):
        """Initialize MUSIQ model"""
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading MUSIQ model on {self.device}...")
        
        self.model = pyiqa.create_metric('musiq', device=self.device)
        
        # Will be set after first pass through data
        self.min_score = None
        self.max_score = None
        
        print("MUSIQ model loaded successfully!")
    
    def predict_raw_score(self, image_path):
        """Get raw MUSIQ score without normalization"""
        try:
            score = self.model(image_path).item()
            return float(score)
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None
    
    def normalize_score(self, raw_score):
        """Normalize score to 0-100 range using learned min/max"""
        if self.min_score is None or self.max_score is None:
            return 50.0  # Default if normalization not set
        
        normalized = ((raw_score - self.min_score) / (self.max_score - self.min_score)) * 100
        return float(np.clip(normalized, 0, 100))
    
    def score_product_groups_adaptive(self, data_structure):
        """
        Two-pass scoring:
        1. First pass: collect raw scores to determine range
        2. Second pass: normalize scores to 0-100
        """
        print("\n=== PASS 1: Collecting raw scores ===")
        
        # First pass: collect all raw scores
        raw_results = []
        raw_scores = []
        
        for category, groups in data_structure.items():
            print(f"Processing category: {category}")
            
            for group, image_paths in groups.items():
                for img_path in tqdm(image_paths, leave=False, desc=f"  {group}"):
                    raw_score = self.predict_raw_score(img_path)
                    if raw_score is not None:
                        raw_results.append({
                            'category': category,
                            'group': group,
                            'image_path': str(img_path),
                            'raw_score': raw_score
                        })
                        raw_scores.append(raw_score)
        
        # Calculate normalization parameters using percentiles
        # Use 5th and 95th percentile to be robust to outliers
        raw_scores = np.array(raw_scores)
        self.min_score = np.percentile(raw_scores, 5)
        self.max_score = np.percentile(raw_scores, 95)
        
        print(f"\n✅ Raw score statistics:")
        print(f"   Min (5th percentile): {self.min_score:.2f}")
        print(f"   Max (95th percentile): {self.max_score:.2f}")
        print(f"   Mean: {np.mean(raw_scores):.2f}")
        print(f"   Std: {np.std(raw_scores):.2f}")
        
        # Second pass: normalize scores
        print("\n=== PASS 2: Normalizing scores ===")
        
        normalized_results = []
        for result in raw_results:
            normalized_score = self.normalize_score(result['raw_score'])
            normalized_results.append({
                'category': result['category'],
                'group': result['group'],
                'image_path': result['image_path'],
                'musiq_score': normalized_score,
                'raw_score': result['raw_score']
            })
        
        df = pd.DataFrame(normalized_results)
        
        print(f"\n✅ Normalized score statistics:")
        print(f"   Min: {df['musiq_score'].min():.2f}")
        print(f"   Max: {df['musiq_score'].max():.2f}")
        print(f"   Mean: {df['musiq_score'].mean():.2f}")
        print(f"   Std: {df['musiq_score'].std():.2f}")
        
        return df
    
    def get_rankings(self, scores, handle_ties=True):
        """Convert scores to rankings"""
        if not scores:
            return []
        
        pairs = [(i, score) for i, score in enumerate(scores)]
        sorted_pairs = sorted(pairs, key=lambda x: x[1], reverse=True)
        
        if handle_ties:
            ranking = [0] * len(scores)
            current_rank = 0
            last_score = None
            
            for i, (idx, score) in enumerate(sorted_pairs):
                if i > 0 and score == last_score:
                    ranking[idx] = current_rank
                else:
                    current_rank = i
                    ranking[idx] = current_rank
                last_score = score
        else:
            ranking = [0] * len(scores)
            for rank, (idx, _) in enumerate(sorted_pairs):
                ranking[idx] = rank
        
        return ranking
    
    def calculate_metrics(self, ground_truth_scores, predicted_scores):
        """Calculate metrics matching your evaluation script"""
        if len(ground_truth_scores) != len(predicted_scores):
            return None
        
        gt = np.array(ground_truth_scores)
        pred = np.array(predicted_scores)
        
        # MSE
        mse = np.mean((gt - pred) ** 2)
        
        # Rankings
        gt_ranking = self.get_rankings(ground_truth_scores)
        pred_ranking = self.get_rankings(predicted_scores)
        
        # Rank Agreement
        rank_agreement = sum(r1 == r2 for r1, r2 in zip(gt_ranking, pred_ranking)) / len(gt_ranking)
        
        # Normalized Ranking Loss
        norm_rank_loss = np.linalg.norm(np.array(gt_ranking) - np.array(pred_ranking)) / len(gt_ranking)
        
        # Top Accuracy
        top_accuracy = 1 if np.argmin(gt_ranking) == np.argmin(pred_ranking) else 0
        
        # Correlations
        spearman_corr, spearman_p = spearmanr(gt, pred)
        kendall_corr, kendall_p = kendalltau(gt, pred)
        pearson_corr = np.corrcoef(gt, pred)[0, 1]
        
        return {
            'mse': float(mse),
            'rank_agreement': float(rank_agreement),
            'normalized_ranking_loss': float(norm_rank_loss),
            'top_accuracy': int(top_accuracy),
            'spearman_correlation': float(spearman_corr),
            'kendall_tau': float(kendall_corr),
            'pearson_correlation': float(pearson_corr)
        }


def build_dataset_structure(dataset_image_path, max_images=4):
    """Build data structure from your dataset format"""
    data_structure = {}
    
    print("Loading dataset structure...")
    for category in os.listdir(dataset_image_path):
        category_path = os.path.join(dataset_image_path, category)
        if not os.path.isdir(category_path):
            continue
        
        data_structure[category] = {}
        
        for group in os.listdir(category_path):
            group_path = os.path.join(dataset_image_path, category, group)
            if not os.path.isdir(group_path):
                continue
            
            images = sorted([
                os.path.join(group_path, img)
                for img in os.listdir(group_path)
                if img.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
            ])
            
            if len(images) > max_images or len(images) == 0:
                continue
            
            data_structure[category][group] = images
    
    print(f"Loaded {len(data_structure)} categories")
    return data_structure


def main():
    """Main function with adaptive normalization"""
    
    # Setup
    dataset_image_path = "/home/debajyoti/paridhi_mtp/NAS/MTP/Persuasive_Image/Persuasive_Image/home/debajyoti/paridhi_mtp/product_images_real/dataset_image_new"
    output_dir = "./musiq_baseline_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize baseline
    baseline = MUSIQAdaptiveBaseline()
    
    # Build dataset structure
    data_structure = build_dataset_structure(dataset_image_path, max_images=4)
    
    # Score all images with adaptive normalization
    df = baseline.score_product_groups_adaptive(data_structure)
    
    # Save results
    output_csv = os.path.join(output_dir, "musiq_scores.csv")
    df.to_csv(output_csv, index=False)
    print(f"\n✅ Scores saved to {output_csv}")
    
    # Compute group-level statistics
    print("\n=== Computing group-level statistics ===")
    
    df['rank_within_group'] = df.groupby(['category', 'group'])['musiq_score'].rank(
        ascending=False, method='first'
    )
    
    group_stats = df.groupby(['category', 'group']).agg({
        'musiq_score': ['mean', 'std', 'min', 'max']
    }).reset_index()
    
    group_stats_file = os.path.join(output_dir, "group_statistics.csv")
    group_stats.to_csv(group_stats_file, index=False)
    print(f"Group statistics saved to {group_stats_file}")
    
    # Display sample results
    print("\n=== Sample Results (First 10 images) ===")
    sample_df = df.head(10)
    print(sample_df[['category', 'group', 'musiq_score', 'rank_within_group']])
    
    print("\n=== Category-Level Statistics ===")
    category_stats = df.groupby('category')['musiq_score'].agg(['mean', 'std', 'min', 'max', 'count'])
    print(category_stats)
    
    print(f"\n✅ MUSIQ baseline complete! Results saved to {output_dir}")
    print(f"\nNext: Run quick_musiq_comparison.py to compare with ground truth")


if __name__ == "__main__":
    main()