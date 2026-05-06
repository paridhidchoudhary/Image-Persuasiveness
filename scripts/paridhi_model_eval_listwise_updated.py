import os
import torch
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from collections import defaultdict
from transformers import AutoProcessor, AutoModelForImageTextToText
from huggingface_hub import login
from qwen_vl_utils import process_vision_info
from scipy.stats import kendalltau, spearmanr
import torch.nn as nn
import random
import json

from simple_data_preprocess import extract_text_data

# ============================================================
# FIX 1: Unified seed — must match training script exactly
# ============================================================
SEED = 48  # was 45 in eval, 50 in training — now unified to match train_test_split seed
random.seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

## SET PATHS
data_root = "home/debajyoti/paridhi_mtp/product_images_real"
dataset_image = os.path.join(data_root, "dataset_image_new")
dataset_response = os.path.join(data_root, "dataset_response_new")
dataset_user_preferred = os.path.join(data_root, "final_data")
output_dir = "./listwise_model_evaluation_fixed"
os.makedirs(output_dir, exist_ok=True)

MAX_IMAGES = 4

MODEL_FILES = {
    "qwen_zeroshot": "output_qwen_zeroshot.txt",
    "qwen_fewshot": "output_qwen_fewshot.txt",
    "pixtral_zeroshot": "output_pixtral_zeroshot.txt",
    "pixtral_fewshot": "output_pixtral_fewshot.txt"
}

FINETUNED_MODEL_NAME = "vlm_finetuned_listwise"


# ============================================================
# Metric helpers (unchanged from original)
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

def calculate_rank_agreement(ranking1, ranking2):
    if not ranking1 or not ranking2 or len(ranking1) != len(ranking2):
        return 0
    agreement = sum(r1 == r2 for r1, r2 in zip(ranking1, ranking2))
    return agreement / len(ranking1)

def calculate_mse(scores1, scores2):
    if not scores1 or not scores2 or len(scores1) != len(scores2):
        return float('inf')
    return np.mean((np.array(scores1) - np.array(scores2)) ** 2)

def calculate_normalized_ranking_loss(ranking1, ranking2):
    if not ranking1 or not ranking2 or len(ranking1) != len(ranking2):
        return float('inf')
    r1 = np.array(ranking1)
    r2 = np.array(ranking2)
    distance = np.linalg.norm(r1 - r2) / len(r1)
    return distance

def calculate_top_accuracy(ranking1, ranking2):
    if not ranking1 or not ranking2 or len(ranking1) != len(ranking2):
        return 0
    top1 = np.argmin(ranking1)
    top2 = np.argmin(ranking2)
    return 1 if top1 == top2 else 0


# ============================================================
# FIX 2: Normalize GT scores to [0,1] before MSE comparison
# This matches how training normalizes scores in collate_fn
# ============================================================

def normalize_scores(scores):
    """
    Normalize scores to [0, 1] range — must match collate_fn in training exactly.
    If all scores are equal, returns 0.5 for all (same as training).
    """
    arr = np.array(scores, dtype=np.float32)
    mn, mx = arr.min(), arr.max()
    if mx > mn:
        return ((arr - mn) / (mx - mn)).tolist()
    else:
        return [0.5] * len(scores)


def extract_scores(file_path):
    try:
        extracted_info, _ = extract_text_data(file_path)
        if extracted_info:
            extracted_info.sort(key=lambda x: x["image_num"])
            scores = [item.get("score", 0) for item in extracted_info]
            if all(score is not None for score in scores):
                return scores, extracted_info
    except Exception as e:
        print(f"Error extracting scores from {file_path}: {e}")
    return None, None


# ============================================================
# FIX 3: Score head kept in float32 — remove .half() call
# In training it was .to(model.device) without half conversion
# ============================================================

def initialize_model():
    print(f"Loading fine-tuned model: {FINETUNED_MODEL_NAME}")
    login(token="hf_xxxxxxxxxx")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(FINETUNED_MODEL_NAME, use_fast=False)

    model = AutoModelForImageTextToText.from_pretrained(FINETUNED_MODEL_NAME).to(device)

    hidden_size = model.config.hidden_size

    # Architecture must match training exactly
    model.score_head = nn.Sequential(
        nn.Linear(hidden_size, hidden_size // 2),
        nn.LayerNorm(hidden_size // 2),
        nn.GELU(),
        nn.Dropout(0.1),
        nn.Linear(hidden_size // 2, 1),
        nn.Sigmoid()
    ).to(device)

    score_head_path = os.path.join(FINETUNED_MODEL_NAME, "score_head.pt")
    if os.path.exists(score_head_path):
        model.score_head.load_state_dict(torch.load(score_head_path, map_location=device))
        print(f"✓ Score head loaded from {score_head_path}")
    else:
        raise FileNotFoundError(f"Score head not found at {score_head_path}!")

    # FIX 3: Do NOT convert score_head to half — keep float32 to match training
    # model.score_head = model.score_head.half()  ← REMOVED

    # Cast score head to match model dtype explicitly
    model_dtype = next(model.parameters()).dtype
    print(f"Model dtype: {model_dtype}")
    if model_dtype == torch.float16:
        # Only cast if model is actually fp16, and match score head input dtype at inference time
        # We'll handle this in get_model_outputs by casting image_embeds before score_head
        pass

    model.eval()
    model.score_head.eval()

    print(f"Model and score head loaded on {device}")
    print(f"Score head dtype: {next(model.score_head.parameters()).dtype}")

    return model, processor, device


def get_image_embeddings_from_vision_tokens(hidden_states, input_ids, num_images):
    vision_start_id = 151652
    vision_end_id = 151653

    input_ids_list = input_ids[0].tolist()
    image_embeddings = []

    i = 0
    while i < len(input_ids_list) and len(image_embeddings) < num_images:
        if input_ids_list[i] == vision_start_id:
            start = i
            j = i + 1
            while j < len(input_ids_list) and input_ids_list[j] != vision_end_id:
                j += 1
            end = j + 1
            image_region_embeds = hidden_states[0, start:end, :]
            pooled_embed = image_region_embeds.mean(dim=0)
            image_embeddings.append(pooled_embed)
            i = end
        else:
            i += 1

    if len(image_embeddings) != num_images:
        print(f"Warning: Found {len(image_embeddings)} vision regions, expected {num_images}")
        return None

    return torch.stack(image_embeddings)


def get_model_outputs(model, processor, device, images, category, group):
    try:
        sorted_images = []
        for img_path in images:
            img = Image.open(img_path).convert("RGB")
            if img.width > 256 or img.height > 256:
                scaling_factor = 256 / float(max(img.width, img.height))
                new_width = int(img.width * scaling_factor)
                new_height = int(img.height * scaling_factor)
                img = img.resize((new_width, new_height), Image.LANCZOS)
            sorted_images.append({"type": "image", "image": img})

        messages = [
            {
                "role": "user",
                "content": sorted_images + [
                    {
                        "type": "text",
                        "text": f"You are evaluating images in the '{group}' group under the '{category}' product category. "
                                f"Rank the images, based on their appeal for selling '{category}' product. "
                                f"Provide description, and *persuasion score (1-100)* for each image and explain the ranking."
                    }
                ],
            }
        ]

        prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = processor(
            text=prompt_text,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            truncation=True,
            max_length=2048,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]

        num_images = len(images)
        image_embeds = get_image_embeddings_from_vision_tokens(
            hidden_states,
            inputs['input_ids'],
            num_images
        )

        if image_embeds is None:
            print(f"Failed to extract image embeddings for {category}/{group}")
            return None, None

        # FIX 3 (cont): Cast image_embeds to float32 before score_head
        # score_head stays float32; hidden states may be float16
        image_embeds_f32 = image_embeds.float()
        pred_scores = model.score_head(image_embeds_f32).squeeze(-1)

        # FIX 2: Return raw [0,1] scores — do NOT scale to 0-100
        # MSE will be computed against normalized GT scores (also [0,1])
        scores_normalized = pred_scores.detach().cpu().numpy().tolist()

        print(f"✓ Scores (normalized [0,1]): {scores_normalized}")

        output_text = f"Predicted scores (normalized [0,1]): {scores_normalized}\n"
        output_file = os.path.join(output_dir, f"{category}_{group}_finetuned_scores.txt")
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(output_text)

        return scores_normalized, output_text

    except Exception as e:
        print(f"Error in get_model_outputs for {category}/{group}: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def load_dataset():
    data = []
    print("Loading dataset...")
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
    print(f"Dataset loaded with {len(data)} groups having ground truth")
    return data


def analyze_model_performance():
    all_data = load_dataset()

    # FIX 1: Use SEED=48 to match training split
    train_data, test_data = train_test_split(all_data, test_size=0.05, random_state=SEED)
    print(f"Validation set size: {len(test_data)} groups")

    # Sanity check — print test categories
    from collections import Counter
    test_cats = Counter(s["category"] for s in test_data)
    print(f"Test set categories: {dict(test_cats)}")

    model, processor, device = initialize_model()

    stats = {
        "total_samples": len(test_data),
        "valid_samples": 0,
        "agreement": {"finetuned_vs_ground_truth": [], "model_vs_ground_truth": defaultdict(list), "finetuned_vs_model": defaultdict(list)},
        "mse": {"finetuned_vs_ground_truth": [], "model_vs_ground_truth": defaultdict(list), "finetuned_vs_model": defaultdict(list)},
        "norm_ranking_loss": {"finetuned_vs_ground_truth": [], "model_vs_ground_truth": defaultdict(list), "finetuned_vs_model": defaultdict(list)},
        "top_accuracy": {"finetuned_vs_ground_truth": [], "model_vs_ground_truth": defaultdict(list), "finetuned_vs_model": defaultdict(list)},
        "kendall_tau": {"finetuned_vs_ground_truth": [], "model_vs_ground_truth": defaultdict(list)},
        "spearman_rho": {"finetuned_vs_ground_truth": [], "model_vs_ground_truth": defaultdict(list)},
        "category_stats": defaultdict(lambda: defaultdict(list)),
        "detailed_results": []
    }

    details_file = os.path.join(output_dir, "detailed_results.txt")
    with open(details_file, "w", encoding="utf-8") as detailed_f:
        detailed_f.write("DETAILED EVALUATION RESULTS (FIXED)\n")
        detailed_f.write("=====================================\n\n")
        detailed_f.write(f"Fixes applied:\n")
        detailed_f.write(f"  1. Seed unified to {SEED} (was 45 in eval, 50 in training)\n")
        detailed_f.write(f"  2. GT scores normalized to [0,1] before MSE (matches training collate_fn)\n")
        detailed_f.write(f"  3. Score head kept float32 (removed .half() conversion)\n\n")

        for i, sample in enumerate(test_data):
            category = sample["category"]
            group = sample["group"]
            images = sample["images"]
            ground_truth_path = sample["ground_truth_path"]

            print(f"\nProcessing {i+1}/{len(test_data)}: {category}/{group}")
            detailed_f.write(f"Sample {i+1}: {category}/{group}\n")

            # Extract raw GT scores
            ground_truth_scores_raw, ground_truth_info = extract_scores(ground_truth_path)
            if not ground_truth_scores_raw:
                print(f"Could not extract ground truth scores")
                detailed_f.write(f"Could not extract ground truth scores\n\n")
                continue

            # FIX 2: Normalize GT scores to [0,1] — same as training collate_fn
            ground_truth_scores = normalize_scores(ground_truth_scores_raw)
            ground_truth_ranking = get_ranking_from_scores(ground_truth_scores, handle_ties=True)

            detailed_f.write(f"  GT scores (raw): {ground_truth_scores_raw}\n")
            detailed_f.write(f"  GT scores (normalized [0,1]): {ground_truth_scores}\n")
            detailed_f.write(f"  GT ranking: {ground_truth_ranking}\n")

            # Get finetuned model scores (already [0,1])
            print(f"  Running fine-tuned model inference...")
            finetuned_scores, finetuned_output = get_model_outputs(model, processor, device, images, category, group)
            
            print(f"Raw pred scores before ranking: {finetuned_scores}")
            print(f"Score range: {max(finetuned_scores) - min(finetuned_scores):.4f}")
            print(f"GT normalized: {ground_truth_scores}")
            if not finetuned_scores:
                print(f"Could not generate scores from fine-tuned model")
                detailed_f.write(f"Could not generate scores from fine-tuned model\n\n")
                continue

            if len(finetuned_scores) != len(ground_truth_scores):
                print(f"Score length mismatch: finetuned={len(finetuned_scores)}, gt={len(ground_truth_scores)}")
                detailed_f.write(f"Score length mismatch\n\n")
                continue

            detailed_f.write(f"  Fine-tuned scores [0,1]: {finetuned_scores}\n")

            finetuned_ranking = get_ranking_from_scores(finetuned_scores, handle_ties=True)
            detailed_f.write(f"  Fine-tuned ranking: {finetuned_ranking}\n")

            # Metrics — both sides now in [0,1]
            agreement = calculate_rank_agreement(ground_truth_ranking, finetuned_ranking)
            mse = calculate_mse(ground_truth_scores, finetuned_scores)  # both [0,1] now
            norm_rank_loss = calculate_normalized_ranking_loss(ground_truth_ranking, finetuned_ranking)
            top_acc = calculate_top_accuracy(ground_truth_ranking, finetuned_ranking)
            kendall = calculate_kendall_tau(ground_truth_ranking, finetuned_ranking)
            spearman = calculate_spearman_rho(ground_truth_ranking, finetuned_ranking)

            stats["agreement"]["finetuned_vs_ground_truth"].append(agreement)
            stats["mse"]["finetuned_vs_ground_truth"].append(mse)
            stats["norm_ranking_loss"]["finetuned_vs_ground_truth"].append(norm_rank_loss)
            stats["top_accuracy"]["finetuned_vs_ground_truth"].append(top_acc)
            stats["kendall_tau"]["finetuned_vs_ground_truth"].append(kendall)
            stats["spearman_rho"]["finetuned_vs_ground_truth"].append(spearman)

            stats["category_stats"][category]["agreement"].append(agreement)
            stats["category_stats"][category]["mse"].append(mse)
            stats["category_stats"][category]["norm_ranking_loss"].append(norm_rank_loss)
            stats["category_stats"][category]["top_accuracy"].append(top_acc)

            detailed_f.write(f"  === Fine-tuned vs GT ===\n")
            detailed_f.write(f"  Agreement: {agreement:.4f}, MSE: {mse:.4f}, NRL: {norm_rank_loss:.4f}, TopAcc: {top_acc}\n")
            detailed_f.write(f"  Kendall: {kendall:.4f}, Spearman: {spearman:.4f}\n")

            # Baselines — also normalize their scores to [0,1] for fair MSE comparison
            model_metrics = {}
            for model_name, file_name in MODEL_FILES.items():
                model_path = os.path.join(dataset_response, category, group, file_name)
                if not os.path.exists(model_path):
                    continue

                model_scores_raw, _ = extract_scores(model_path)
                if not model_scores_raw:
                    continue
                if len(model_scores_raw) != len(ground_truth_scores_raw):
                    continue

                # FIX 2: Normalize baseline scores too for MSE consistency
                model_scores = normalize_scores(model_scores_raw)
                model_ranking = get_ranking_from_scores(model_scores, handle_ties=True)

                gt_agreement = calculate_rank_agreement(ground_truth_ranking, model_ranking)
                gt_mse = calculate_mse(ground_truth_scores, model_scores)
                gt_norm_rank_loss = calculate_normalized_ranking_loss(ground_truth_ranking, model_ranking)
                gt_top_acc = calculate_top_accuracy(ground_truth_ranking, model_ranking)
                gt_kendall = calculate_kendall_tau(ground_truth_ranking, model_ranking)
                gt_spearman = calculate_spearman_rho(ground_truth_ranking, model_ranking)

                ft_agreement = calculate_rank_agreement(finetuned_ranking, model_ranking)
                ft_mse = calculate_mse(finetuned_scores, model_scores)
                ft_norm_rank_loss = calculate_normalized_ranking_loss(finetuned_ranking, model_ranking)
                ft_top_acc = calculate_top_accuracy(finetuned_ranking, model_ranking)
                ft_kendall = calculate_kendall_tau(finetuned_ranking, model_ranking)
                ft_spearman = calculate_spearman_rho(finetuned_ranking, model_ranking)

                stats["agreement"]["model_vs_ground_truth"][model_name].append(gt_agreement)
                stats["mse"]["model_vs_ground_truth"][model_name].append(gt_mse)
                stats["norm_ranking_loss"]["model_vs_ground_truth"][model_name].append(gt_norm_rank_loss)
                stats["top_accuracy"]["model_vs_ground_truth"][model_name].append(gt_top_acc)
                stats["kendall_tau"]["model_vs_ground_truth"][model_name].append(gt_kendall)
                stats["spearman_rho"]["model_vs_ground_truth"][model_name].append(gt_spearman)

                stats["agreement"]["finetuned_vs_model"][model_name].append(ft_agreement)
                stats["mse"]["finetuned_vs_model"][model_name].append(ft_mse)
                stats["norm_ranking_loss"]["finetuned_vs_model"][model_name].append(ft_norm_rank_loss)
                stats["top_accuracy"]["finetuned_vs_model"][model_name].append(ft_top_acc)

                model_metrics[model_name] = {
                    "scores_normalized": model_scores,
                    "ranking": model_ranking,
                    "vs_ground_truth": {
                        "agreement": gt_agreement, "mse": gt_mse,
                        "norm_ranking_loss": gt_norm_rank_loss, "top_accuracy": gt_top_acc,
                        "kendall_tau": gt_kendall, "spearman_rho": gt_spearman,
                    }
                }

                detailed_f.write(f"  {model_name}: agree={gt_agreement:.3f}, mse={gt_mse:.4f}, nrl={gt_norm_rank_loss:.3f}, top={gt_top_acc}\n")

            stats["valid_samples"] += 1
            stats["detailed_results"].append({
                "category": category,
                "group": group,
                "ground_truth_scores_raw": ground_truth_scores_raw,
                "ground_truth_scores_normalized": ground_truth_scores,
                "ground_truth_ranking": ground_truth_ranking,
                "finetuned_scores": finetuned_scores,
                "finetuned_ranking": finetuned_ranking,
                "metrics": {
                    "finetuned_vs_ground_truth": {
                        "agreement": agreement, "mse": mse,
                        "norm_ranking_loss": norm_rank_loss, "top_accuracy": top_acc,
                        "kendall_tau": kendall, "spearman_rho": spearman,
                    },
                    "model_metrics": model_metrics
                }
            })
            detailed_f.write("\n" + "-"*60 + "\n\n")

    # Aggregate results
    def safe_mean(lst): return float(np.mean(lst)) if lst else 0.0

    results = {
        "total_samples": stats["total_samples"],
        "valid_samples": stats["valid_samples"],
        "fixes_applied": [
            f"Seed unified to {SEED}",
            "GT and baseline scores normalized to [0,1] before MSE (matches training)",
            "Score head kept float32 (removed .half())"
        ],
        "avg_agreement": {
            "finetuned_vs_ground_truth": safe_mean(stats["agreement"]["finetuned_vs_ground_truth"]),
            "model_vs_ground_truth": {m: safe_mean(v) for m, v in stats["agreement"]["model_vs_ground_truth"].items()},
            "finetuned_vs_model": {m: safe_mean(v) for m, v in stats["agreement"]["finetuned_vs_model"].items()},
        },
        "avg_mse": {
            "finetuned_vs_ground_truth": safe_mean(stats["mse"]["finetuned_vs_ground_truth"]),
            "model_vs_ground_truth": {m: safe_mean(v) for m, v in stats["mse"]["model_vs_ground_truth"].items()},
            "finetuned_vs_model": {m: safe_mean(v) for m, v in stats["mse"]["finetuned_vs_model"].items()},
        },
        "avg_norm_ranking_loss": {
            "finetuned_vs_ground_truth": safe_mean(stats["norm_ranking_loss"]["finetuned_vs_ground_truth"]),
            "model_vs_ground_truth": {m: safe_mean(v) for m, v in stats["norm_ranking_loss"]["model_vs_ground_truth"].items()},
            "finetuned_vs_model": {m: safe_mean(v) for m, v in stats["norm_ranking_loss"]["finetuned_vs_model"].items()},
        },
        "avg_top_accuracy": {
            "finetuned_vs_ground_truth": safe_mean(stats["top_accuracy"]["finetuned_vs_ground_truth"]),
            "model_vs_ground_truth": {m: safe_mean(v) for m, v in stats["top_accuracy"]["model_vs_ground_truth"].items()},
            "finetuned_vs_model": {m: safe_mean(v) for m, v in stats["top_accuracy"]["finetuned_vs_model"].items()},
        },
        "avg_kendall_tau": {
            "finetuned_vs_ground_truth": safe_mean(stats["kendall_tau"]["finetuned_vs_ground_truth"]),
            "model_vs_ground_truth": {m: safe_mean(v) for m, v in stats["kendall_tau"]["model_vs_ground_truth"].items()},
        },
        "avg_spearman_rho": {
            "finetuned_vs_ground_truth": safe_mean(stats["spearman_rho"]["finetuned_vs_ground_truth"]),
            "model_vs_ground_truth": {m: safe_mean(v) for m, v in stats["spearman_rho"]["model_vs_ground_truth"].items()},
        },
        "category_metrics": {
            cat: {
                "agreement": safe_mean(m["agreement"]),
                "mse": safe_mean(m["mse"]),
                "norm_ranking_loss": safe_mean(m["norm_ranking_loss"]),
                "top_accuracy": safe_mean(m["top_accuracy"]),
            }
            for cat, m in stats["category_stats"].items()
        }
    }

    return results, stats


if __name__ == "__main__":
    print("Running FIXED comprehensive model evaluation...")
    print("Fixes: (1) seed=48, (2) normalized MSE, (3) float32 score head\n")

    results, raw_stats = analyze_model_performance()

    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer): return int(obj)
            if isinstance(obj, np.floating): return float(obj)
            if isinstance(obj, np.ndarray): return obj.tolist()
            return super().default(obj)

    with open(os.path.join(output_dir, "comprehensive_evaluation_results.json"), "w") as f:
        json.dump(results, f, indent=2, cls=NpEncoder)

    with open(os.path.join(output_dir, "detailed_stats.json"), "w") as f:
        json.dump({"detailed_results": raw_stats["detailed_results"]}, f, indent=2, cls=NpEncoder)

    print("\n===== FIXED EVALUATION SUMMARY =====")
    print(f"Fixes applied: {results['fixes_applied']}")
    print(f"\nTotal: {results['total_samples']}  |  Valid: {results['valid_samples']}")

    print("\n----- Agreement -----")
    print(f"Fine-tuned: {results['avg_agreement']['finetuned_vs_ground_truth']:.4f}")
    for m, v in results['avg_agreement']['model_vs_ground_truth'].items():
        print(f"{m}: {v:.4f}")

    print("\n----- MSE (normalized [0,1]) -----")
    print(f"Fine-tuned: {results['avg_mse']['finetuned_vs_ground_truth']:.4f}")
    for m, v in results['avg_mse']['model_vs_ground_truth'].items():
        print(f"{m}: {v:.4f}")

    print("\n----- Normalized Ranking Loss -----")
    print(f"Fine-tuned: {results['avg_norm_ranking_loss']['finetuned_vs_ground_truth']:.4f}")
    for m, v in results['avg_norm_ranking_loss']['model_vs_ground_truth'].items():
        print(f"{m}: {v:.4f}")

    print("\n----- Top Accuracy -----")
    print(f"Fine-tuned: {results['avg_top_accuracy']['finetuned_vs_ground_truth']:.4f}")
    for m, v in results['avg_top_accuracy']['model_vs_ground_truth'].items():
        print(f"{m}: {v:.4f}")

    print("\n----- Kendall / Spearman -----")
    print(f"Fine-tuned: Kendall={results['avg_kendall_tau']['finetuned_vs_ground_truth']:.4f}  Spearman={results['avg_spearman_rho']['finetuned_vs_ground_truth']:.4f}")
    for m in results['avg_kendall_tau']['model_vs_ground_truth']:
        print(f"{m}: Kendall={results['avg_kendall_tau']['model_vs_ground_truth'][m]:.4f}  Spearman={results['avg_spearman_rho']['model_vs_ground_truth'][m]:.4f}")

    print("\n----- Category-wise (Fine-tuned) -----")
    for cat, m in results["category_metrics"].items():
        print(f"{cat.upper()}: agree={m['agreement']:.3f}, mse={m['mse']:.4f}, nrl={m['norm_ranking_loss']:.3f}, top={m['top_accuracy']:.3f}")

    print(f"\nResults saved to {output_dir}/")