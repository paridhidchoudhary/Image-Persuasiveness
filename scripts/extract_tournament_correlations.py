"""
Extract Spearman and Kendall correlations from Tournament-based evaluation
Handles the pairwise/tournament structure differently than full-context evaluation
"""

import json
import pandas as pd
import numpy as np
from scipy.stats import spearmanr, kendalltau

def calculate_correlations_from_tournament(json_file):
    """
    Calculate correlations from tournament evaluation results
    
    Args:
        json_file: Path to tournament results JSON
    
    Returns:
        Dictionary with correlation metrics
    """
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    detailed_results = data['detailed_results']
    
    # Storage for all score pairs
    all_gt_scores = []
    all_pred_scores = []
    
    # Storage for per-group correlations
    group_correlations = {
        'category': [],
        'group': [],
        'spearman': [],
        'kendall': [],
        'num_images': []
    }
    
    print("="*70)
    print("EXTRACTING CORRELATIONS FROM TOURNAMENT EVALUATION")
    print("="*70)
    print(f"\nTotal groups in evaluation: {len(detailed_results)}")
    
    skipped = 0
    valid = 0
    
    for result in detailed_results:
        category = result['category']
        group = result['group']
        
        # Get ground truth scores (for all images in group)
        gt_scores = result['ground_truth_scores']
        
        # For tournament, we only have finalist scores
        # Need to reconstruct full predicted scores
        finalist_scores = result['predicted_scores_finalists']
        finalist_indices = result['finalist_indices']
        
        # If we have all images scored, use ground truth length
        # Otherwise, we can only use finalist pairs
        if len(gt_scores) == 2:
            # Direct 2-image comparison
            pred_scores = finalist_scores
            
        elif len(gt_scores) > 2:
            # Tournament structure - only finalists are scored
            # For correlation, we can only use the finalist scores
            # Extract corresponding GT scores
            gt_scores_finalists = [gt_scores[i] for i in finalist_indices]
            gt_scores = gt_scores_finalists
            pred_scores = finalist_scores
        else:
            skipped += 1
            continue
        
        # Check for constant arrays (which make correlation undefined)
        if len(set(gt_scores)) <= 1 or len(set(pred_scores)) <= 1:
            skipped += 1
            continue
        
        # Calculate correlations for this group
        try:
            spear, _ = spearmanr(gt_scores, pred_scores)
            kend, _ = kendalltau(gt_scores, pred_scores)
            
            if np.isnan(spear) or np.isnan(kend):
                skipped += 1
                continue
            
            # Store group-level correlations
            group_correlations['category'].append(category)
            group_correlations['group'].append(group)
            group_correlations['spearman'].append(spear)
            group_correlations['kendall'].append(kend)
            group_correlations['num_images'].append(len(gt_scores))
            
            # Add to global lists
            all_gt_scores.extend(gt_scores)
            all_pred_scores.extend(pred_scores)
            
            valid += 1
            
        except Exception as e:
            print(f"Error processing {category}/{group}: {e}")
            skipped += 1
            continue
    
    print(f"\nValid groups for correlation: {valid}")
    print(f"Skipped groups: {skipped}")
    
    # Calculate overall correlations
    overall_spearman = np.mean(group_correlations['spearman'])
    overall_kendall = np.mean(group_correlations['kendall'])
    
    # Also calculate global correlation (across all score pairs)
    global_spearman, _ = spearmanr(all_gt_scores, all_pred_scores)
    global_kendall, _ = kendalltau(all_gt_scores, all_pred_scores)
    
    print("\n" + "="*70)
    print("CORRELATION RESULTS")
    print("="*70)
    print(f"\n{'Metric':<30} {'Per-Group Avg':>20} {'Global':>15}")
    print("-"*70)
    print(f"{'Spearman ρ':<30} {overall_spearman:>20.4f} {global_spearman:>15.4f}")
    print(f"{'Kendall τ':<30} {overall_kendall:>20.4f} {global_kendall:>15.4f}")
    print(f"{'Sample Size':<30} {valid:>20} {len(all_gt_scores):>15}")
    print("="*70)
    
    # Category-wise breakdown
    print("\n" + "="*70)
    print("CATEGORY-WISE CORRELATIONS")
    print("="*70)
    
    df = pd.DataFrame(group_correlations)
    category_stats = df.groupby('category').agg({
        'spearman': ['mean', 'std', 'count'],
        'kendall': ['mean', 'std']
    }).round(4)
    
    print(category_stats)
    
    # Save detailed results
    df.to_csv('tournament_correlations_detailed_fulltext.csv', index=False)
    print(f"\n✅ Detailed results saved to: tournament_correlations_detailed_fulltext.csv")
    
    # Return summary
    return {
        'overall_spearman_avg': overall_spearman,
        'overall_kendall_avg': overall_kendall,
        'global_spearman': global_spearman,
        'global_kendall': global_kendall,
        'valid_groups': valid,
        'total_score_pairs': len(all_gt_scores),
        'category_breakdown': category_stats
    }


def compare_with_musiq(tournament_results):
    """
    Compare tournament results with MUSIQ baseline
    """
    print("\n" + "="*70)
    print("COMPARISON WITH BASELINES")
    print("="*70)
    
    # Try to load MUSIQ results
    try:
        musiq_df = pd.read_csv('musiq_eval_validation_only.csv')
        musiq_valid = musiq_df[~musiq_df['spearman_correlation'].isna()]
        
        musiq_spearman = musiq_valid['spearman_correlation'].mean()
        musiq_kendall = musiq_valid['kendall_tau'].mean()
        
        print(f"\n{'Model':<30} {'Spearman ρ':>15} {'Kendall τ':>15} {'n':>10}")
        print("-"*70)
        print(f"{'Your Tournament Model':<30} {tournament_results['overall_spearman_avg']:>15.4f} "
              f"{tournament_results['overall_kendall_avg']:>15.4f} {tournament_results['valid_groups']:>10}")
        print(f"{'MUSIQ (ICCV 2021)':<30} {musiq_spearman:>15.4f} "
              f"{musiq_kendall:>15.4f} {len(musiq_valid):>10}")
        
        # Calculate improvement
        spear_improvement = ((tournament_results['overall_spearman_avg'] - musiq_spearman) / 
                            abs(musiq_spearman)) * 100
        kend_improvement = ((tournament_results['overall_kendall_avg'] - musiq_kendall) / 
                           abs(musiq_kendall)) * 100
        
        print("\n" + "-"*70)
        print("IMPROVEMENT OVER MUSIQ:")
        print(f"  Spearman: {spear_improvement:+.1f}%")
        print(f"  Kendall:  {kend_improvement:+.1f}%")
        
    except Exception as e:
        print(f"\nNote: Could not load MUSIQ results for comparison - {e}")
    
    print("="*70)


def main():
    """
    Main function to process tournament evaluation
    """
    
    # Path to your tournament results
    #tournament_file = "/home/debajyoti/paridhi_mtp/NAS/MTP/Persuasive_Image/Persuasive_Image/pairwise_evaluation_results_pairwise_2/evaluation_results.json"
    tournament_file = "/home/debajyoti/paridhi_mtp/NAS/MTP/Persuasive_Image/Persuasive_Image/pairwise_evaluation_results_fullcontext_2/evaluation_results.json"
    print(f"Loading tournament results from: {tournament_file}\n")
    
    try:
        results = calculate_correlations_from_tournament(tournament_file)
        
        # Compare with MUSIQ if available
        compare_with_musiq(results)
        
        # Create summary for report
        print("\n" + "="*70)
        print("SUMMARY FOR YOUR REPORT")
        print("="*70)
        print(f"""
Tournament-Based Evaluation Results:
- Spearman's ρ: {results['overall_spearman_avg']:.4f}
- Kendall's τ:  {results['overall_kendall_avg']:.4f}
- Valid groups: {results['valid_groups']}
- Score pairs:  {results['total_score_pairs']}

These metrics measure how well your model's pairwise rankings
correlate with ground truth persuasiveness scores.
        """)
        
        return results
        
    except FileNotFoundError:
        print(f"❌ ERROR: Could not find file: {tournament_file}")
        print("\nPlease update the tournament_file path to match your file location:")
        print("  - Check your evaluation output directory")
        print("  - Look for files like 'tournament_results.json' or 'pairwise_results.json'")
        return None


if __name__ == "__main__":
    main()