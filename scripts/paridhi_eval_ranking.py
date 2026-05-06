import os
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
output_dir = "./listwise_model_evaluation_v2"
os.makedirs(output_dir, exist_ok=True)

FINETUNED_MODEL_NAME = "vlm_stage2_ranking"
TEST_IDS_FILE = "vlm_finetuned_ranking_v2_test_ids.json"
MAX_IMAGES = 4

MODEL_FILES = {
    "qwen_zeroshot": "output_qwen_zeroshot.txt",
    "qwen_fewshot": "output_qwen_fewshot.txt",
    "pixtral_zeroshot": "output_pixtral_zeroshot.txt",
    "pixtral_fewshot": "output_pixtral_fewshot.txt"
}

# ============================================================
# Metric helpers
# ============================================================

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


# ============================================================
# Model loading — score head stays float32
# ============================================================

def initialize_model():
    print(f"Loading: {FINETUNED_MODEL_NAME}")
    login(token="hf_xxxxxxxxx")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(FINETUNED_MODEL_NAME, use_fast=False)
    model = AutoModelForImageTextToText.from_pretrained(FINETUNED_MODEL_NAME).to(device)

    hidden_size = model.config.hidden_size
    model.score_head = nn.Sequential(
        nn.Linear(hidden_size, hidden_size // 2),
        nn.LayerNorm(hidden_size // 2),
        nn.GELU(),
        nn.Dropout(0.1),
        nn.Linear(hidden_size // 2, 1),
        nn.Sigmoid()
    ).to(device)  # float32, no .half()

    score_head_path = os.path.join(FINETUNED_MODEL_NAME, "score_head.pt")
    if os.path.exists(score_head_path):
        model.score_head.load_state_dict(torch.load(score_head_path, map_location=device))
        print(f"✓ Score head loaded")
    else:
        raise FileNotFoundError(f"Score head not found at {score_head_path}")

    model.eval()
    model.score_head.eval()
    print(f"Model ready on {device}")
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
    if len(embeddings) != num_images:
        return None
    return torch.stack(embeddings)


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

        prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = processor(
            text=prompt_text, images=image_inputs, videos=video_inputs,
            padding=True, truncation=True, max_length=2048, return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]

        image_embeds = get_image_embeddings(hidden_states, inputs['input_ids'], len(images))
        if image_embeds is None:
            return None, None

        # float32 for score head
        pred_scores = model.score_head(image_embeds.float()).squeeze(-1)
        scores = pred_scores.detach().cpu().numpy().tolist()
        print(f"  Scores [0,1]: {[round(s,4) for s in scores]} | range: {max(scores)-min(scores):.4f}")
        return scores, None

    except Exception as e:
        print(f"Error in inference {category}/{group}: {e}")
        import traceback
        traceback.print_exc()
        return None, None


# ============================================================
# Dataset — load all, then filter to saved test IDs
# ============================================================

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
    """Load exact test set used during training via saved IDs."""
    all_data = load_dataset()

    if os.path.exists(TEST_IDS_FILE):
        print(f"✓ Loading test IDs from {TEST_IDS_FILE}")
        with open(TEST_IDS_FILE) as f:
            test_ids = json.load(f)
        test_set = {(d["category"], d["group"]) for d in test_ids}
        test_data = [d for d in all_data if (d["category"], d["group"]) in test_set]
        print(f"Matched {len(test_data)}/{len(test_ids)} test groups")
    else:
        # Fallback: reproduce split with same stratified function
        print(f"⚠️  {TEST_IDS_FILE} not found, reproducing split")
        from collections import defaultdict

        def stratified_split(data, test_ratio=0.10, min_test=2, seed=SEED):
            rng = random.Random(seed)
            by_cat = defaultdict(list)
            for item in data:
                by_cat[item['category']].append(item)
            train, test = [], []
            for cat, items in by_cat.items():
                items = items.copy()
                rng.shuffle(items)
                n_test = max(min_test, math.ceil(len(items) * test_ratio))
                n_test = min(n_test, len(items) - 1)
                test.extend(items[:n_test])
                train.extend(items[n_test:])
            return train, test

        _, test_data = stratified_split(all_data, test_ratio=0.10, min_test=2)

    # Print category coverage
    from collections import Counter
    cats = Counter(d["category"] for d in test_data)
    print(f"Test set category coverage: {dict(sorted(cats.items()))}")
    return test_data


# ============================================================
# Main evaluation
# ============================================================

def analyze_model_performance():
    test_data = get_test_data()
    print(f"\nEvaluating on {len(test_data)} test groups")

    model, processor, device = initialize_model()

    stats = {
        "total_samples": len(test_data),
        "valid_samples": 0,
        "agreement": {"ft_vs_gt": [], "model_vs_gt": defaultdict(list)},
        "mse": {"ft_vs_gt": [], "model_vs_gt": defaultdict(list)},
        "nrl": {"ft_vs_gt": [], "model_vs_gt": defaultdict(list)},
        "top_acc": {"ft_vs_gt": [], "model_vs_gt": defaultdict(list)},
        "kendall": {"ft_vs_gt": [], "model_vs_gt": defaultdict(list)},
        "spearman": {"ft_vs_gt": [], "model_vs_gt": defaultdict(list)},
        "category": defaultdict(lambda: defaultdict(list)),
        "detailed": []
    }

    details_file = os.path.join(output_dir, "detailed_results.txt")
    with open(details_file, "w") as f:
        f.write("EVALUATION RESULTS — v2 (ranking-focused training)\n")
        f.write("=" * 60 + "\n\n")

        for i, sample in enumerate(test_data):
            category = sample["category"]
            group = sample["group"]
            images = sample["images"]

            print(f"\n[{i+1}/{len(test_data)}] {category}/{group} ({len(images)} images)")
            f.write(f"\n[{i+1}] {category}/{group}\n")

            # GT scores — normalized to [0,1] for MSE
            gt_raw, _ = extract_scores(sample["ground_truth_path"])
            if not gt_raw:
                f.write("  SKIP: no GT scores\n")
                continue

            gt_norm = normalize_scores(gt_raw)
            gt_ranking = get_ranking_from_scores(gt_norm, handle_ties=True)
            f.write(f"  GT raw: {gt_raw}\n")
            f.write(f"  GT norm: {[round(s,3) for s in gt_norm]}\n")
            f.write(f"  GT ranking: {gt_ranking}\n")

            # Fine-tuned model
            ft_scores, _ = get_model_outputs(model, processor, device, images, category, group)
            if not ft_scores or len(ft_scores) != len(gt_norm):
                f.write("  SKIP: inference failed\n")
                continue

            ft_ranking = get_ranking_from_scores(ft_scores, handle_ties=True)
            f.write(f"  FT scores: {[round(s,3) for s in ft_scores]}\n")
            f.write(f"  FT ranking: {ft_ranking}\n")

            agree = calculate_rank_agreement(gt_ranking, ft_ranking)
            mse = calculate_mse(gt_norm, ft_scores)
            nrl = calculate_normalized_ranking_loss(gt_ranking, ft_ranking)
            top = calculate_top_accuracy(gt_ranking, ft_ranking)
            kend = calculate_kendall_tau(gt_ranking, ft_ranking)
            spear = calculate_spearman_rho(gt_ranking, ft_ranking)

            stats["agreement"]["ft_vs_gt"].append(agree)
            stats["mse"]["ft_vs_gt"].append(mse)
            stats["nrl"]["ft_vs_gt"].append(nrl)
            stats["top_acc"]["ft_vs_gt"].append(top)
            stats["kendall"]["ft_vs_gt"].append(kend)
            stats["spearman"]["ft_vs_gt"].append(spear)

            for k, v in [("agreement", agree), ("mse", mse), ("nrl", nrl), ("top_acc", top)]:
                stats["category"][category][k].append(v)

            f.write(f"  FT metrics: agree={agree:.3f}, mse={mse:.4f}, nrl={nrl:.3f}, top={top}, K={kend:.3f}, S={spear:.3f}\n")

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

                stats["agreement"]["model_vs_gt"][mname].append(calculate_rank_agreement(gt_ranking, m_ranking))
                stats["mse"]["model_vs_gt"][mname].append(calculate_mse(gt_norm, m_norm))
                stats["nrl"]["model_vs_gt"][mname].append(calculate_normalized_ranking_loss(gt_ranking, m_ranking))
                stats["top_acc"]["model_vs_gt"][mname].append(calculate_top_accuracy(gt_ranking, m_ranking))
                stats["kendall"]["model_vs_gt"][mname].append(calculate_kendall_tau(gt_ranking, m_ranking))
                stats["spearman"]["model_vs_gt"][mname].append(calculate_spearman_rho(gt_ranking, m_ranking))

                f.write(f"  {mname}: agree={calculate_rank_agreement(gt_ranking, m_ranking):.3f}, "
                        f"K={calculate_kendall_tau(gt_ranking, m_ranking):.3f}\n")

            stats["valid_samples"] += 1
            stats["detailed"].append({
                "category": category, "group": group,
                "gt_raw": gt_raw, "gt_norm": gt_norm, "gt_ranking": gt_ranking,
                "ft_scores": ft_scores, "ft_ranking": ft_ranking,
                "metrics": {"agree": agree, "mse": mse, "nrl": nrl, "top": top, "kendall": kend, "spearman": spear}
            })

    def sm(lst): return float(np.mean(lst)) if lst else 0.0

    results = {
        "total_samples": stats["total_samples"],
        "valid_samples": stats["valid_samples"],
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
    print("=" * 60)
    print("EVALUATION v2 — Ranking-focused fine-tuned model")
    print("=" * 60)

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

    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Total: {results['total_samples']}  |  Valid: {results['valid_samples']}")

    metrics = [
        ("Agreement", "avg_agreement", True),
        ("MSE (normalized)", "avg_mse", False),
        ("NRL", "avg_norm_ranking_loss", False),
        ("Top Accuracy", "avg_top_accuracy", True),
        ("Kendall Tau", "avg_kendall_tau", True),
        ("Spearman Rho", "avg_spearman_rho", True),
    ]

    for label, key, higher_better in metrics:
        print(f"\n----- {label} ({'↑' if higher_better else '↓'}) -----")
        ft_val = results[key]["finetuned_vs_ground_truth"]
        print(f"  Fine-tuned: {ft_val:.4f}")
        best_baseline = None
        best_val = None
        for m, v in results[key].get("model_vs_ground_truth", {}).items():
            marker = ""
            if best_val is None or (higher_better and v > best_val) or (not higher_better and v < best_val):
                best_val = v
                best_baseline = m
            print(f"  {m}: {v:.4f}")
        if best_baseline and best_val is not None:
            if higher_better:
                diff = ((ft_val - best_val) / abs(best_val)) * 100 if best_val != 0 else 0
                marker = f"  → FT {'beats' if diff > 0 else 'trails'} best baseline by {abs(diff):.1f}%"
            else:
                diff = ((best_val - ft_val) / abs(best_val)) * 100 if best_val != 0 else 0
                marker = f"  → FT {'beats' if diff > 0 else 'trails'} best baseline by {abs(diff):.1f}%"
            print(marker)

    print(f"\n----- Category-wise (Fine-tuned) -----")
    for cat, m in sorted(results["category_metrics"].items()):
        print(f"  {cat.upper():15s}: agree={m.get('agreement',0):.3f}, "
              f"nrl={m.get('nrl',0):.3f}, top={m.get('top_acc',0):.3f}")

    print(f"\nResults saved to {output_dir}/")