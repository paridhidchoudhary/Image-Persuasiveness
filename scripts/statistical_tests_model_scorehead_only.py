"""
Statistical significance testing for ranking model evaluation.

Now includes:
- VLM baselines (qwen, pixtral)
- MUSIQ + NIMA (external models with per_sample results)

Tests:
- Wilcoxon signed-rank
- Bootstrap CI
- Permutation test
- McNemar (top_accuracy)
"""

import json
import os
import numpy as np
from scipy.stats import wilcoxon
from statsmodels.stats.contingency_tables import mcnemar

# ── FILE PATHS ────────────────────────────────────────────────────────────────
STATS_FILE = "./listwise_model_evaluation_scorehead_only_lora_ranking_with_explanations/detailed_stats.json"
OUTPUT_DIR = "./listwise_model_evaluation_scorehead_only_lora_ranking_with_explanations"

MUSIQ_FILE = "/home/debajyoti/paridhi_mtp/nas_final/MTP/Persuasive_Image/Persuasive_Image/MUSIQ/musiq_baseline_results_2/musiq_summary.json"
NIMA_FILE  = "./nima_baseline_results/nima_summary.json"

ALPHA = 0.05

BASELINES = ["qwen_zeroshot", "qwen_fewshot", "pixtral_zeroshot", "pixtral_fewshot"]
EXTRA_BASELINES = ["musiq", "nima"]

CONTINUOUS_METRICS = [
    ("kendall_tau", True,  "Kendall's Tau"),
    ("spearman_rho", True,  "Spearman's Rho"),
    ("agreement", True,  "Agreement"),
    ("norm_ranking_loss", False, "Norm Ranking Loss *"),
]

# ── LOAD MAIN RESULTS ─────────────────────────────────────────────────────────
with open(STATS_FILE) as f:
    data = json.load(f)

results = data["detailed_results"]
print(f"Loaded {len(results)} test samples\n")

# ── LOAD MUSIQ / NIMA ─────────────────────────────────────────────────────────
def load_external(file_path, key):
    with open(file_path) as f:
        data = json.load(f)
    return data[key]["per_sample"]

musiq_data = load_external(MUSIQ_FILE, "musiq")
nima_data  = load_external(NIMA_FILE, "nima")

# ── HELPERS ───────────────────────────────────────────────────────────────────

def extract_paired_internal(results, baseline_name):
    out = {m[0]: ([], []) for m in CONTINUOUS_METRICS}
    out["top_accuracy"] = ([], [])

    for r in results:
        mm = r["metrics"]["model_metrics"]
        ft = r["metrics"]["finetuned_vs_ground_truth"]

        if baseline_name not in mm:
            continue

        bl = mm[baseline_name]["vs_ground_truth"]

        for metric, _, _ in CONTINUOUS_METRICS:
            if metric in ft and metric in bl:
                out[metric][0].append(ft[metric])
                out[metric][1].append(bl[metric])

        if "top_accuracy" in ft and "top_accuracy" in bl:
            out["top_accuracy"][0].append(ft["top_accuracy"])
            out["top_accuracy"][1].append(bl["top_accuracy"])

    return out


def extract_paired_external(results, external_data):
    out = {m[0]: ([], []) for m in CONTINUOUS_METRICS}
    out["top_accuracy"] = ([], [])

    for r in results:
        key = f"{r['category']}/{r['group']}"

        if key not in external_data:
            continue

        ft = r["metrics"]["finetuned_vs_ground_truth"]
        ext = external_data[key]

        for metric, _, _ in CONTINUOUS_METRICS:
            if metric in ft and metric in ext:
                out[metric][0].append(ft[metric])
                out[metric][1].append(ext[metric])

        if "top_accuracy" in ft and "top_accuracy" in ext:
            out["top_accuracy"][0].append(ft["top_accuracy"])
            out["top_accuracy"][1].append(ext["top_accuracy"])

    return out


# ── STAT TESTS ────────────────────────────────────────────────────────────────

def run_wilcoxon(ft, bl, higher_better):
    ft, bl = np.array(ft), np.array(bl)
    diffs = ft - bl

    if np.all(diffs == 0):
        return {"p_value": 1.0, "mean_diff": 0.0, "significant": False}

    alt = "greater" if higher_better else "less"
    stat, p = wilcoxon(ft, bl, alternative=alt)

    return {
        "p_value": float(p),
        "mean_diff": float(np.mean(diffs)),
        "ft_mean": float(np.mean(ft)),
        "bl_mean": float(np.mean(bl)),
        "significant": p < ALPHA
    }


def bootstrap_ci(ft, bl, n_boot=10000):
    diffs = np.array(ft) - np.array(bl)
    n = len(diffs)

    boot = []
    for _ in range(n_boot):
        idx = np.random.randint(0, n, n)
        boot.append(np.mean(diffs[idx]))

    ci_low = np.percentile(boot, 2.5)
    ci_high = np.percentile(boot, 97.5)

    return {
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "significant": not (ci_low <= 0 <= ci_high)
    }


def permutation_test(ft, bl, higher_better, n_perm=10000):
    ft = np.array(ft)
    bl = np.array(bl)

    observed = np.mean(ft - bl)
    count = 0

    for _ in range(n_perm):
        swap = np.random.rand(len(ft)) < 0.5
        new_ft = np.where(swap, bl, ft)
        new_bl = np.where(swap, ft, bl)

        diff = np.mean(new_ft - new_bl)

        if higher_better:
            count += (diff >= observed)
        else:
            count += (diff <= observed)

    return {"p_value": count / n_perm}


def run_mcnemar(ft, bl):
    ft = np.array(ft)
    bl = np.array(bl)

    br = np.sum((ft == 1) & (bl == 1))
    fo = np.sum((ft == 1) & (bl == 0))
    bo = np.sum((ft == 0) & (bl == 1))
    bw = np.sum((ft == 0) & (bl == 0))

    res = mcnemar([[br, fo], [bo, bw]], exact=(fo + bo < 25))

    return {"p_value": float(res.pvalue)}

def convert_numpy(obj):
    if isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(v) for v in obj]
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    else:
        return obj

# ── MAIN LOOP ────────────────────────────────────────────────────────────────

ALL_BASELINES = BASELINES + EXTRA_BASELINES
all_results = {}

print("=" * 72)
print("STATISTICAL SIGNIFICANCE TESTS")
print("=" * 72)

for baseline in ALL_BASELINES:

    if baseline in BASELINES:
        paired = extract_paired_internal(results, baseline)
    elif baseline == "musiq":
        paired = extract_paired_external(results, musiq_data)
    elif baseline == "nima":
        paired = extract_paired_external(results, nima_data)

    print(f"\n{'─'*72}")
    print(f"  Fine-tuned vs {baseline}")
    print(f"{'─'*72}")

    br = {}

    for metric, higher_better, label in CONTINUOUS_METRICS:
        ft_v, bl_v = paired[metric]

        if len(ft_v) < 5:
            continue

        res = run_wilcoxon(ft_v, bl_v, higher_better)
        boot = bootstrap_ci(ft_v, bl_v)
        perm = permutation_test(ft_v, bl_v, higher_better)

        print(f"{label:<25} Δ={res['mean_diff']:+.4f}  p={res['p_value']:.4f}")
        print(f"  CI → [{boot['ci_low']:+.4f}, {boot['ci_high']:+.4f}]")
        print(f"  perm p → {perm['p_value']:.4f}")

        br[metric] = {
            **res,
            "bootstrap": boot,
            "permutation": perm
        }

    # McNemar
    ft_ta, bl_ta = paired["top_accuracy"]
    if len(ft_ta) >= 5:
        mc = run_mcnemar(ft_ta, bl_ta)
        print(f"top_accuracy McNemar p={mc['p_value']:.4f}")
        br["top_accuracy"] = mc

    all_results[baseline] = br


# ── SAVE ─────────────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)
out_path = os.path.join(OUTPUT_DIR, "statistical_test_results_all.json")

clean_results = convert_numpy(all_results)

with open(out_path, "w") as f:
    json.dump(clean_results, f, indent=2)
print(f"\nSaved to {out_path}")