import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
import numpy as np
from PIL import Image
from collections import defaultdict
import torch.nn as nn
import random
import json
import math
from transformers import AutoProcessor, AutoModelForImageTextToText
from huggingface_hub import login
from qwen_vl_utils import process_vision_info
from scipy.stats import kendalltau, spearmanr

from simple_data_preprocess import extract_text_data

# ── Config ────────────────────────────────────────────────────
SEED = 48
random.seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

data_root = "home/debajyoti/paridhi_mtp/product_images_real"
dataset_image = os.path.join(data_root, "dataset_image_new")
dataset_response = os.path.join(data_root, "dataset_response_new")
dataset_user_preferred = os.path.join(data_root, "final_data")
output_dir = "./listwise_model_evaluation_stage2"
os.makedirs(output_dir, exist_ok=True)

# Two-stage model paths
STAGE1_MODEL = "vlm_stage1_lm"                          # LM weights from Stage 1
STAGE2_SCORE_HEAD = "vlm_stage2_ranking/score_head.pt"  # Score head from Stage 2
TEST_IDS_FILE = "test_ids.json"                         # Saved by train_stage1_lm.py
MAX_IMAGES = 4

MODEL_FILES = {
    "qwen_zeroshot": "output_qwen_zeroshot.txt",
    "qwen_fewshot": "output_qwen_fewshot.txt",
    "pixtral_zeroshot": "output_pixtral_zeroshot.txt",
    "pixtral_fewshot": "output_pixtral_fewshot.txt"
}

# ── Metrics ───────────────────────────────────────────────────

def calculate_kendall_tau(ranking1, ranking2):
    if not ranking1 or not ranking2 or len(ranking1) != len(ranking2):
        return 0.0
    tau, _ = kendalltau(ranking1, ranking2)
    return 0.0 if np.isnan(tau) else float(tau)

def calculate_spearman_rho(ranking1, ranking2):
    if not ranking1 or not ranking2 or len(ranking1) != len(ranking2):
        return 0.0
    rho, _ = spearmanr(ranking1, ranking2)
    return 0.0 if np.isnan(rho) else float(rho)

def get_ranking_from_scores(scores, handle_ties=True):
    if not scores:
        return []
    pairs = [(i, s) for i, s in enumerate(scores)]
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

def calculate_rank_agreement(r1, r2):
    if not r1 or not r2 or len(r1) != len(r2): return 0
    return sum(a == b for a, b in zip(r1, r2)) / len(r1)

def calculate_mse(s1, s2):
    if not s1 or not s2 or len(s1) != len(s2): return float('inf')
    return float(np.mean((np.array(s1) - np.array(s2)) ** 2))

def calculate_normalized_ranking_loss(r1, r2):
    if not r1 or not r2 or len(r1) != len(r2): return float('inf')
    return float(np.linalg.norm(np.array(r1) - np.array(r2)) / len(r1))

def calculate_top_accuracy(r1, r2):
    if not r1 or not r2 or len(r1) != len(r2): return 0
    return 1 if np.argmin(r1) == np.argmin(r2) else 0

def normalize_scores(scores):
    """Normalize to [0,1] — matches training collate_fn exactly."""
    arr = np.array(scores, dtype=np.float32)
    mn, mx = arr.min(), arr.max()
    if mx > mn:
        return ((arr - mn) / (mx - mn)).tolist()
    return [0.5] * len(scores)

def extract_scores(file_path):
    try:
        extracted_info, _ = extract_text_data(file_path)
        if extracted_info:
            extracted_info.sort(key=lambda x: x["image_num"])
            scores = [item.get("score", 0) for item in extracted_info]
            if all(s is not None for s in scores):
                return scores, extracted_info
    except Exception as e:
        print(f"Error extracting scores from {file_path}: {e}")
    return None, None


# ── Model loading ─────────────────────────────────────────────

def initialize_model():
    print(f"Loading Stage 1 LM from: {STAGE1_MODEL}")
    print(f"Loading Stage 2 score head from: {STAGE2_SCORE_HEAD}")

    login(token="hf_xxxxxxxxxxxxxxxxxxxxx")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load processor from Stage 1
    processor = AutoProcessor.from_pretrained(STAGE1_MODEL, use_fast=False)

    # Load LM from Stage 1 (fine-tuned for structured score generation)
    model = AutoModelForImageTextToText.from_pretrained(
        STAGE1_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    # Recreate score head — must match Stage 2 training architecture exactly
    hidden_size = model.config.hidden_size
    model.score_head = nn.Sequential(
        nn.Linear(hidden_size, hidden_size // 2),
        nn.LayerNorm(hidden_size // 2),
        nn.GELU(),
        nn.Dropout(0.1),
        nn.Linear(hidden_size // 2, 1),
        nn.Sigmoid()
    ).to("cuda:0")  # float32, explicitly on GPU

    # Load Stage 2 score head weights
    if not os.path.exists(STAGE2_SCORE_HEAD):
        raise FileNotFoundError(f"Score head not found at {STAGE2_SCORE_HEAD}")

    model.score_head.load_state_dict(
        torch.load(STAGE2_SCORE_HEAD, map_location="cuda:0")
    )
    print(f"✓ Score head loaded from {STAGE2_SCORE_HEAD}")

    model.eval()
    model.score_head.eval()

    print(f"Model ready on {device}")
    print(f"LM dtype: {next(model.parameters()).dtype}")
    print(f"Score head dtype: {next(model.score_head.parameters()).dtype}")
    return model, processor, device


def get_image_embeddings(hidden_states, input_ids, num_images):
    vision_start_id = 151652
    vision_end_id = 151653
    ids = input_ids[0].tolist()
    embeddings = []
    i = 0
    while i < len(ids) and len(embeddings) < num_images:
        if ids[i] == vision_start_id:
            start = i
            j = i + 1
            while j < len(ids) and ids[j] != vision_end_id:
                j += 1
            end = j + 1
            region = hidden_states[0, start:end, :]
            embeddings.append(region.mean(dim=0))
            i = end
        else:
            i += 1
    if len(embeddings) == 0:
        return None
    # Return however many we found (partial is ok)
    return torch.stack(embeddings), len(embeddings)


def get_model_outputs(model, processor, device, images, category, group):
    try:
        sorted_images = []
        for img_path in images:
            img = Image.open(img_path).convert("RGB")
            if img.width > 256 or img.height > 256:
                factor = 256 / float(max(img.width, img.height))
                img = img.resize((int(img.width * factor), int(img.height * factor)), Image.LANCZOS)
            sorted_images.append({"type": "image", "image": img})

        messages = [{
            "role": "user",
            "content": sorted_images + [{
                "type": "text",
                "text": f"You are evaluating images in the '{group}' group under the '{category}' product category. "
                        f"Rank the images, based on their appeal for selling '{category}' product. "
                        f"Provide description, and *persuasion score (1-100)* for each image and explain the ranking."
            }]
        }]

        prompt_text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = processor(
            text=prompt_text,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            truncation=True,
            max_length=2048,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]

        result = get_image_embeddings(hidden_states, inputs['input_ids'], len(images))
        if result is None:
            print(f"  No embeddings extracted for {category}/{group}")
            return None, None

        image_embeds, found = result
        if found != len(images):
            print(f"  Partial embeddings: {found}/{len(images)} for {category}/{group}")

        # float32 for score head — hidden states are float16
        pred_scores = model.score_head(image_embeds.float()).squeeze(-1)
        scores = pred_scores.detach().cpu().numpy().tolist()

        # Ensure it's a list even for single image
        if not isinstance(scores, list):
            scores = [scores]

        print(f"  Scores [0,1]: {[round(s, 4) for s in scores]} | range: {max(scores)-min(scores):.4f}")
        return scores, None

    except Exception as e:
        print(f"  Error in inference {category}/{group}: {e}")
        import traceback
        traceback.print_exc()
        return None, None


# ── Dataset loading ───────────────────────────────────────────

def load_dataset():
    data = []
    for category in os.listdir(dataset_image):
        cat_path = os.path.join(dataset_image, category)
        if not os.path.isdir(cat_path):
            continue
        for group in os.listdir(cat_path):
            group_path = os.path.join(cat_path, group)
            if not os.path.isdir(group_path):
                continue
            images = sorted([
                os.path.join(group_path, img)
                for img in os.listdir(group_path)
                if img.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
            ])
            if len(images) > MAX_IMAGES or len(images) == 0:
                continue
            gt_path = os.path.join(dataset_user_preferred, category, group, "user_output.txt")
            if os.path.exists(gt_path):
                data.append({
                    "images": images,
                    "category": category,
                    "group": group,
                    "ground_truth_path": gt_path
                })
    return data


def get_test_data():
    """Load exact test set saved by train_stage1_lm.py."""
    all_data = load_dataset()

    if not os.path.exists(TEST_IDS_FILE):
        raise FileNotFoundError(
            f"{TEST_IDS_FILE} not found. "
            f"Make sure train_stage1_lm.py ran successfully and saved test_ids.json."
        )

    with open(TEST_IDS_FILE) as f:
        test_ids = json.load(f)

    test_set = {(d["category"], d["group"]) for d in test_ids}
    test_data = [d for d in all_data if (d["category"], d["group"]) in test_set]

    print(f"Test IDs in file: {len(test_ids)}")
    print(f"Matched in dataset: {len(test_data)}")

    from collections import Counter
    cats = Counter(d["category"] for d in test_data)
    print(f"Categories: {dict(sorted(cats.items()))}")

    missing_cats = set(d["category"] for d in all_data) - set(cats.keys())
    if missing_cats:
        print(f"⚠️  Missing categories: {missing_cats}")
    else:
        print("✓ All categories present in test set")

    return test_data


# ── Main evaluation ───────────────────────────────────────────

def analyze_model_performance():
    test_data = get_test_data()
    print(f"\nEvaluating on {len(test_data)} test groups\n")

    model, processor, device = initialize_model()

    stats = {
        "total_samples": len(test_data),
        "valid_samples": 0,
        "agreement": {"ft_vs_gt": [], "model_vs_gt": defaultdict(list)},
        "mse":        {"ft_vs_gt": [], "model_vs_gt": defaultdict(list)},
        "nrl":        {"ft_vs_gt": [], "model_vs_gt": defaultdict(list)},
        "top_acc":    {"ft_vs_gt": [], "model_vs_gt": defaultdict(list)},
        "kendall":    {"ft_vs_gt": [], "model_vs_gt": defaultdict(list)},
        "spearman":   {"ft_vs_gt": [], "model_vs_gt": defaultdict(list)},
        "category":   defaultdict(lambda: defaultdict(list)),
        "detailed":   []
    }

    details_file = os.path.join(output_dir, "detailed_results.txt")
    with open(details_file, "w") as f:
        f.write("EVALUATION — Two-Stage Model (Stage1 LM + Stage2 Score Head)\n")
        f.write("=" * 70 + "\n\n")

        for i, sample in enumerate(test_data):
            category = sample["category"]
            group = sample["group"]
            images = sample["images"]

            print(f"[{i+1}/{len(test_data)}] {category}/{group} ({len(images)} images)")
            f.write(f"\n[{i+1}] {category}/{group}\n")

            # GT
            gt_raw, _ = extract_scores(sample["ground_truth_path"])
            if not gt_raw:
                f.write("  SKIP: no GT scores\n")
                continue

            gt_norm = normalize_scores(gt_raw)
            gt_ranking = get_ranking_from_scores(gt_norm, handle_ties=True)
            f.write(f"  GT raw: {gt_raw}\n")
            f.write(f"  GT norm: {[round(s,3) for s in gt_norm]}\n")
            f.write(f"  GT ranking: {gt_ranking}\n")

            # Fine-tuned model inference
            ft_scores, _ = get_model_outputs(model, processor, device, images, category, group)
            if not ft_scores:
                f.write("  SKIP: inference failed\n")
                continue

            # Handle partial embeddings — trim GT to match
            if len(ft_scores) != len(gt_norm):
                print(f"  Score count mismatch: ft={len(ft_scores)}, gt={len(gt_norm)} — trimming GT")
                gt_norm = gt_norm[:len(ft_scores)]
                gt_raw = gt_raw[:len(ft_scores)]
                gt_ranking = get_ranking_from_scores(gt_norm, handle_ties=True)

            ft_ranking = get_ranking_from_scores(ft_scores, handle_ties=True)
            f.write(f"  FT scores: {[round(s,3) for s in ft_scores]}\n")
            f.write(f"  FT ranking: {ft_ranking}\n")

            agree = calculate_rank_agreement(gt_ranking, ft_ranking)
            mse   = calculate_mse(gt_norm, ft_scores)
            nrl   = calculate_normalized_ranking_loss(gt_ranking, ft_ranking)
            top   = calculate_top_accuracy(gt_ranking, ft_ranking)
            kend  = calculate_kendall_tau(gt_ranking, ft_ranking)
            spear = calculate_spearman_rho(gt_ranking, ft_ranking)

            stats["agreement"]["ft_vs_gt"].append(agree)
            stats["mse"]["ft_vs_gt"].append(mse)
            stats["nrl"]["ft_vs_gt"].append(nrl)
            stats["top_acc"]["ft_vs_gt"].append(top)
            stats["kendall"]["ft_vs_gt"].append(kend)
            stats["spearman"]["ft_vs_gt"].append(spear)

            for k, v in [("agreement", agree), ("mse", mse), ("nrl", nrl), ("top_acc", top)]:
                stats["category"][category][k].append(v)

            f.write(f"  Metrics: agree={agree:.3f}, mse={mse:.4f}, nrl={nrl:.3f}, "
                    f"top={top}, K={kend:.3f}, S={spear:.3f}\n")

            # Baselines
            for mname, fname in MODEL_FILES.items():
                mpath = os.path.join(dataset_response, category, group, fname)
                if not os.path.exists(mpath):
                    continue
                m_raw, _ = extract_scores(mpath)
                if not m_raw or len(m_raw) != len(gt_raw):
                    continue

                m_norm = normalize_scores(m_raw)
                m_ranking = get_ranking_from_scores(m_norm, handle_ties=True)

                stats["agreement"]["model_vs_gt"][mname].append(
                    calculate_rank_agreement(gt_ranking, m_ranking))
                stats["mse"]["model_vs_gt"][mname].append(
                    calculate_mse(gt_norm, m_norm))
                stats["nrl"]["model_vs_gt"][mname].append(
                    calculate_normalized_ranking_loss(gt_ranking, m_ranking))
                stats["top_acc"]["model_vs_gt"][mname].append(
                    calculate_top_accuracy(gt_ranking, m_ranking))
                stats["kendall"]["model_vs_gt"][mname].append(
                    calculate_kendall_tau(gt_ranking, m_ranking))
                stats["spearman"]["model_vs_gt"][mname].append(
                    calculate_spearman_rho(gt_ranking, m_ranking))

                f.write(f"  {mname}: agree={calculate_rank_agreement(gt_ranking, m_ranking):.3f}, "
                        f"K={calculate_kendall_tau(gt_ranking, m_ranking):.3f}\n")

            stats["valid_samples"] += 1
            stats["detailed"].append({
                "category": category, "group": group,
                "gt_raw": gt_raw, "gt_norm": gt_norm, "gt_ranking": gt_ranking,
                "ft_scores": ft_scores, "ft_ranking": ft_ranking,
                "metrics": {
                    "agree": agree, "mse": mse, "nrl": nrl,
                    "top": top, "kendall": kend, "spearman": spear
                }
            })

    def sm(lst): return float(np.mean(lst)) if lst else 0.0

    results = {
        "total_samples": stats["total_samples"],
        "valid_samples": stats["valid_samples"],
        "model_info": {
            "stage1_lm": STAGE1_MODEL,
            "stage2_score_head": STAGE2_SCORE_HEAD,
            "test_ids_file": TEST_IDS_FILE,
        },
        "avg_agreement": {
            "finetuned_vs_ground_truth": sm(stats["agreement"]["ft_vs_gt"]),
            "model_vs_ground_truth": {m: sm(v) for m, v in stats["agreement"]["model_vs_gt"].items()}
        },
        "avg_mse": {
            "finetuned_vs_ground_truth": sm(stats["mse"]["ft_vs_gt"]),
            "model_vs_ground_truth": {m: sm(v) for m, v in stats["mse"]["model_vs_gt"].items()}
        },
        "avg_norm_ranking_loss": {
            "finetuned_vs_ground_truth": sm(stats["nrl"]["ft_vs_gt"]),
            "model_vs_ground_truth": {m: sm(v) for m, v in stats["nrl"]["model_vs_gt"].items()}
        },
        "avg_top_accuracy": {
            "finetuned_vs_ground_truth": sm(stats["top_acc"]["ft_vs_gt"]),
            "model_vs_ground_truth": {m: sm(v) for m, v in stats["top_acc"]["model_vs_gt"].items()}
        },
        "avg_kendall_tau": {
            "finetuned_vs_ground_truth": sm(stats["kendall"]["ft_vs_gt"]),
            "model_vs_ground_truth": {m: sm(v) for m, v in stats["kendall"]["model_vs_gt"].items()}
        },
        "avg_spearman_rho": {
            "finetuned_vs_ground_truth": sm(stats["spearman"]["ft_vs_gt"]),
            "model_vs_ground_truth": {m: sm(v) for m, v in stats["spearman"]["model_vs_gt"].items()}
        },
        "category_metrics": {
            cat: {k: sm(v) for k, v in m.items()}
            for cat, m in stats["category"].items()
        }
    }
    return results, stats


if __name__ == "__main__":
    print("=" * 70)
    print("EVALUATION — Two-Stage Model")
    print(f"  LM:         {STAGE1_MODEL}")
    print(f"  Score Head: {STAGE2_SCORE_HEAD}")
    print(f"  Test IDs:   {TEST_IDS_FILE}")
    print("=" * 70)

    results, raw_stats = analyze_model_performance()

    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer): return int(obj)
            if isinstance(obj, np.floating): return float(obj)
            if isinstance(obj, np.ndarray): return obj.tolist()
            return super().default(obj)

    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2, cls=NpEncoder)

    with open(os.path.join(output_dir, "detailed_stats.json"), "w") as f:
        json.dump({"detailed": raw_stats["detailed"]}, f, indent=2, cls=NpEncoder)

    print(f"\n{'='*70}")
    print("RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"Total: {results['total_samples']}  |  Valid: {results['valid_samples']}")

    metrics = [
        ("Agreement",       "avg_agreement",          True),
        ("MSE (normalized)","avg_mse",                 False),
        ("NRL",             "avg_norm_ranking_loss",   False),
        ("Top Accuracy",    "avg_top_accuracy",        True),
        ("Kendall Tau",     "avg_kendall_tau",         True),
        ("Spearman Rho",    "avg_spearman_rho",        True),
    ]

    for label, key, higher_better in metrics:
        print(f"\n----- {label} ({'↑' if higher_better else '↓'}) -----")
        ft_val = results[key]["finetuned_vs_ground_truth"]
        baselines = results[key].get("model_vs_ground_truth", {})
        print(f"  Fine-tuned (Ours): {ft_val:.4f}")
        best_val = None
        best_name = None
        for m, v in baselines.items():
            print(f"  {m}: {v:.4f}")
            if best_val is None or (higher_better and v > best_val) or (not higher_better and v < best_val):
                best_val = v
                best_name = m
        if best_name:
            diff = ((ft_val - best_val) / abs(best_val)) * 100 if best_val != 0 else 0
            if not higher_better:
                diff = -diff
            status = "✅ BEATS" if diff > 0 else "❌ trails"
            print(f"  → {status} best baseline ({best_name}) by {abs(diff):.1f}%")

    print(f"\n----- Category-wise (Fine-tuned) -----")
    for cat, m in sorted(results["category_metrics"].items()):
        print(f"  {cat.upper():15s}: agree={m.get('agreement',0):.3f}, "
              f"nrl={m.get('nrl',0):.3f}, top={m.get('top_acc',0):.3f}")

    print(f"\nResults saved to {output_dir}/")