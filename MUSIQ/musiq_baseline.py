"""
MUSIQ Baseline for Product Image Persuasiveness Evaluation
Paper: "MUSIQ: Multi-scale Image Quality Transformer" (ICCV 2021)
Install: pip install pyiqa torch torchvision pillow pandas numpy scipy tqdm
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

class MUSIQBaseline:
    def __init__(self, device=None):
        """
        Initialize MUSIQ model using pyiqa library
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading MUSIQ model on {self.device}...")
        
        # Load MUSIQ model (pretrained on KonIQ-10k)
        # This returns scores in [0, 1] range (will be scaled to [0, 100])
        self.model = pyiqa.create_metric('musiq', device=self.device)
        
        print("MUSIQ model loaded successfully!")
    
    def predict_score(self, image_path):
        """
        Predict MUSIQ quality score for a single image
        Returns: float score (0-100 scale)
        """
        try:
            # MUSIQ expects PIL Image or tensor
            score = self.model(image_path).item()
            
            # Scale from [0,1] to [0,100] to match your persuasion scores
            scaled_score = score * 100
            
            return scaled_score
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return 50.0  # Return neutral score on error
    
    def score_product_groups(self, data_structure):
        """
        Score product image groups matching your eval script format
        
        Args:
            data_structure: Dict like your dataset structure:
            {
                'category': {
                    'group': [image_paths]
                }
            }
        
        Returns:
            DataFrame with columns: category, group, image_path, musiq_score
        """
        results = []
        
        for category, groups in data_structure.items():
            print(f"\nProcessing category: {category}")
            
            for group, image_paths in groups.items():
                print(f"  Group: {group} ({len(image_paths)} images)")
                
                for img_path in tqdm(image_paths, leave=False, desc=f"  {group}"):
                    score = self.predict_score(img_path)
                    results.append({
                        'category': category,
                        'group': group,
                        'image_path': str(img_path),
                        'musiq_score': score
                    })
        
        df = pd.DataFrame(results)
        return df
    
    def get_rankings(self, scores, handle_ties=True):
        """
        Convert scores to rankings (matches your evaluation script)
        Higher score = better rank (rank 0 is best)
        """
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
        """
        Calculate metrics matching your evaluation script
        Returns: dict with MSE, rank_agreement, spearman, kendall, top_accuracy
        """
        if len(ground_truth_scores) != len(predicted_scores):
            return None
        
        # Convert to numpy arrays
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
        
        # Correlation metrics
        spearman_corr, spearman_p = spearmanr(gt, pred)
        kendall_corr, kendall_p = kendalltau(gt, pred)
        pearson_corr = np.corrcoef(gt, pred)[0, 1]
        
        return {
            'mse': float(mse),
            'rank_agreement': float(rank_agreement),
            'normalized_ranking_loss': float(norm_rank_loss),
            'top_accuracy': int(top_accuracy),
            'spearman_correlation': float(spearman_corr),
            'spearman_pvalue': float(spearman_p),
            'kendall_tau': float(kendall_corr),
            'kendall_pvalue': float(kendall_p),
            'pearson_correlation': float(pearson_corr)
        }


def build_dataset_structure(dataset_image_path, max_images=4):
    """
    Build data structure from your dataset format
    Matches your load_dataset() function
    """
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
    """
    Main function to generate MUSIQ baseline results
    Matches your evaluation workflow
    """
    
    # === SETUP ===
    # Update these paths to match your setup
    dataset_image_path = "/home/debajyoti/paridhi_mtp/NAS/MTP/Persuasive_Image/Persuasive_Image/home/debajyoti/paridhi_mtp/product_images_real/dataset_image_new"
    output_dir = "./musiq_baseline_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize MUSIQ
    baseline = MUSIQBaseline()
    
    # Build dataset structure
    data_structure = build_dataset_structure(dataset_image_path, max_images=4)
    
    # === SCORE ALL IMAGES ===
    print("\n=== Scoring images with MUSIQ ===")
    df = baseline.score_product_groups(data_structure)
    
    # Save results
    output_csv = os.path.join(output_dir, "musiq_scores.csv")
    df.to_csv(output_csv, index=False)
    print(f"\nScores saved to {output_csv}")
    
    # === COMPUTE GROUP-LEVEL STATISTICS ===
    print("\n=== Computing group-level statistics ===")
    
    # Add rankings within each group
    df['rank_within_group'] = df.groupby(['category', 'group'])['musiq_score'].rank(
        ascending=False, method='first'
    )
    
    # Compute group statistics
    group_stats = df.groupby(['category', 'group']).agg({
        'musiq_score': ['mean', 'std', 'min', 'max']
    }).reset_index()
    
    group_stats_file = os.path.join(output_dir, "group_statistics.csv")
    group_stats.to_csv(group_stats_file, index=False)
    print(f"Group statistics saved to {group_stats_file}")
    
    # === DISPLAY SAMPLE RESULTS ===
    print("\n=== Sample Results ===")
    sample_category = list(data_structure.keys())[0]
    sample_df = df[df['category'] == sample_category].head(10)
    print(sample_df[['category', 'group', 'musiq_score', 'rank_within_group']])
    
    print("\n=== Category-Level Statistics ===")
    category_stats = df.groupby('category')['musiq_score'].agg(['mean', 'std', 'count'])
    print(category_stats)
    
    print(f"\n✅ MUSIQ baseline complete! Results saved to {output_dir}")
    print(f"\nNext step: Compare with your VLM scores using the metrics from your eval script")


if __name__ == "__main__":
    main()