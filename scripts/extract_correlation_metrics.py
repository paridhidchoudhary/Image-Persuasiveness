"""
Extract Spearman and Kendall metrics from your evaluation results
Run this on your comprehensive_evaluation_results.json
"""

import json
import pandas as pd
import numpy as np
from scipy.stats import spearmanr, kendalltau
# Load your evaluation results
#results_file = "./comprehensive_model_evaluation_pairwise_2/detailed_stats.json"
results_file = "./comprehensive_model_evaluation_fulltext_ft_3/detailed_stats.json"

with open(results_file, 'r') as f:
    data = json.load(f)

# Extract Spearman and Kendall for each model
detailed_results = data['detailed_results']

# Storage for correlation metrics
model_correlations = {
    'finetuned': {'spearman': [], 'kendall': []},
    'qwen_zeroshot': {'spearman': [], 'kendall': []},
    'qwen_fewshot': {'spearman': [], 'kendall': []},
    'pixtral_zeroshot': {'spearman': [], 'kendall': []},
    'pixtral_fewshot': {'spearman': [], 'kendall': []}
}

# Process each result
skipped_constant = 0
total_samples = len(detailed_results)

for result in detailed_results:
    # Fine-tuned model (calculated from finetuned_scores vs ground_truth_scores)
    from scipy.stats import spearmanr, kendalltau
    
    gt_scores = result['ground_truth_scores']
    ft_scores = result['finetuned_scores']
    
    # Calculate correlations for fine-tuned model
    if len(gt_scores) == len(ft_scores):
        # Check for constant arrays (which make correlation undefined)
        if len(set(gt_scores)) > 1 and len(set(ft_scores)) > 1:
            spear, _ = spearmanr(gt_scores, ft_scores)
            kend, _ = kendalltau(gt_scores, ft_scores)
            # Only add if not nan
            if not np.isnan(spear) and not np.isnan(kend):
                model_correlations['finetuned']['spearman'].append(spear)
                model_correlations['finetuned']['kendall'].append(kend)
        else:
            skipped_constant += 1
    
    # Other models
    if 'model_metrics' in result['metrics']:
        for model_name, metrics in result['metrics']['model_metrics'].items():
            if model_name in model_correlations:
                # Calculate from scores
                model_scores = metrics['scores']
                if len(gt_scores) == len(model_scores):
                    # Check for constant arrays
                    if len(set(gt_scores)) > 1 and len(set(model_scores)) > 1:
                        spear, _ = spearmanr(gt_scores, model_scores)
                        kend, _ = kendalltau(gt_scores, model_scores)
                        # Only add if not nan
                        if not np.isnan(spear) and not np.isnan(kend):
                            model_correlations[model_name]['spearman'].append(spear)
                            model_correlations[model_name]['kendall'].append(kend)

# Calculate averages
import numpy as np

print("="*70)
print("DIAGNOSTIC INFO")
print("="*70)
print(f"Total validation samples: {total_samples}")
print(f"Samples skipped (constant scores): {skipped_constant}")
print(f"Valid samples for correlation: {total_samples - skipped_constant}")
print("="*70)

print("\n" + "="*70)
print("SPEARMAN'S RHO AND KENDALL'S TAU (Validation Set)")
print("="*70)
print(f"\n{'Model':<25} {'Spearman ρ':>15} {'Kendall τ':>15} {'n':>10}")
print("-"*70)

summary_data = []

for model_name, metrics in model_correlations.items():
    if metrics['spearman']:
        avg_spearman = np.mean(metrics['spearman'])
        avg_kendall = np.mean(metrics['kendall'])
        n_samples = len(metrics['spearman'])
        
        display_name = {
            'finetuned': 'Your Fine-tuned Model',
            'qwen_zeroshot': 'Qwen Zero-shot',
            'qwen_fewshot': 'Qwen Few-shot',
            'pixtral_zeroshot': 'Pixtral Zero-shot',
            'pixtral_fewshot': 'Pixtral Few-shot'
        }[model_name]
        
        print(f"{display_name:<25} {avg_spearman:>15.4f} {avg_kendall:>15.4f} {n_samples:>10}")
        
        summary_data.append({
            'model': display_name,
            'spearman': avg_spearman,
            'kendall': avg_kendall,
            'n': n_samples
        })

print("="*70)

# Also add MUSIQ
musiq_file = "musiq_eval_validation_only.csv"
try:
    musiq_df = pd.read_csv(musiq_file)
    # Filter out nan values
    musiq_valid = musiq_df[~musiq_df['spearman_correlation'].isna()]
    
    if len(musiq_valid) > 0:
        musiq_spearman = musiq_valid['spearman_correlation'].mean()
        musiq_kendall = musiq_valid['kendall_tau'].mean()
        
        print(f"\n{'MUSIQ (ICCV 2021)':<25} {musiq_spearman:>15.4f} {musiq_kendall:>15.4f} {len(musiq_valid):>10}")
        
        summary_data.append({
            'model': 'MUSIQ (ICCV 2021)',
            'spearman': musiq_spearman,
            'kendall': musiq_kendall,
            'n': len(musiq_valid)
        })
except Exception as e:
    print(f"\nNote: MUSIQ results not found - {e}")

# Save to CSV
summary_df = pd.DataFrame(summary_data)
summary_df.to_csv('correlation_metrics_summary_fullcontext_fulltext.csv', index=False)
print(f"\n✅ Saved: correlation_metrics_summary_fullcontext_fulltext.csv")

# Ranking analysis
print("\n" + "="*70)
print("RANKING BY CORRELATION METRICS")
print("="*70)

print("\nBy Spearman's ρ (higher is better):")
sorted_spear = sorted(summary_data, key=lambda x: x['spearman'], reverse=True)
for i, item in enumerate(sorted_spear, 1):
    print(f"{i}. {item['model']:<30} {item['spearman']:.4f}")

print("\nBy Kendall's τ (higher is better):")
sorted_kend = sorted(summary_data, key=lambda x: x['kendall'], reverse=True)
for i, item in enumerate(sorted_kend, 1):
    print(f"{i}. {item['model']:<30} {item['kendall']:.4f}")

print("="*70)