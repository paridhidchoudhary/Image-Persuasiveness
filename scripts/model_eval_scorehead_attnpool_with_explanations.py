import os
import torch
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from collections import defaultdict
from huggingface_hub import login
from qwen_vl_utils import process_vision_info
from scipy.stats import kendalltau, spearmanr
import torch.nn as nn
from transformers import AutoProcessor
import random
import numpy as np

from simple_data_preprocess import extract_text_data

SEED = 50
random.seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

# ── Paths ──────────────────────────────────────────────────────────────────────
data_root = "home/debajyoti/paridhi_mtp/product_images_real"
dataset_image          = os.path.join(data_root, "dataset_image_new")
dataset_response       = os.path.join(data_root, "dataset_response_new")
dataset_user_preferred = os.path.join(data_root, "final_data")
#FINETUNED_MODEL_NAME   = "vlm_finetuned_lora_ranking"
#output_dir             = "./listwise_model_evaluation_scorehead_only_lora_ranking_with_explanations"
FINETUNED_MODEL_NAME   = "vlm_finetuned_attn_pooling"
output_dir             = "./listwise_model_evaluation_attn_pooling"
os.makedirs(output_dir, exist_ok=True)

# Directory to save per-sample explanation text files
explanations_dir = os.path.join(output_dir, "explanations")
os.makedirs(explanations_dir, exist_ok=True)

MODEL_FILES = {
    "qwen_zeroshot":    "output_qwen_zeroshot.txt",
    "qwen_fewshot":     "output_qwen_fewshot.txt",
    "pixtral_zeroshot": "output_pixtral_zeroshot.txt",
    "pixtral_fewshot":  "output_pixtral_fewshot.txt",
}

class AttentionPooling(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Linear(hidden_size, 1, bias=True)
 
    def forward(self, image_region):
        attn_scores  = self.attention(image_region)
        attn_weights = torch.softmax(attn_scores, dim=0)
        pooled       = (attn_weights * image_region).sum(dim=0)
        return pooled

# ── Metric helpers (unchanged) ─────────────────────────────────────────────────
def calculate_kendall_tau(scores1, scores2):
    if not scores1 or not scores2 or len(scores1) != len(scores2):
        return 0.0
    tau, _ = kendalltau(scores1, scores2)
    return 0.0 if np.isnan(tau) else float(tau)

def calculate_spearman_rho(scores1, scores2):
    if not scores1 or not scores2 or len(scores1) != len(scores2):
        return 0.0
    rho, _ = spearmanr(scores1, scores2)
    return 0.0 if np.isnan(rho) else float(rho)

def get_ranking_from_scores(scores, handle_ties=True):
    if not scores:
        return []
    pairs = [(i, s) for i, s in enumerate(scores)]
    sorted_pairs = sorted(pairs, key=lambda x: x[1], reverse=True)
    ranking = [0] * len(scores)
    if handle_ties:
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
        for rank, (idx, _) in enumerate(sorted_pairs):
            ranking[idx] = rank
    return ranking

def calculate_rank_agreement(r1, r2):
    if not r1 or not r2 or len(r1) != len(r2):
        return 0.0
    return sum(a == b for a, b in zip(r1, r2)) / len(r1)

def calculate_mse(s1, s2):
    if not s1 or not s2 or len(s1) != len(s2):
        return float('inf')
    return float(np.mean((np.array(s1) - np.array(s2)) ** 2))

def calculate_normalized_ranking_loss(r1, r2):
    if not r1 or not r2 or len(r1) != len(r2):
        return float('inf')
    return float(np.linalg.norm(np.array(r1) - np.array(r2)) / len(r1))

def calculate_top_accuracy(r1, r2):
    if not r1 or not r2 or len(r1) != len(r2):
        return 0
    return 1 if np.argmin(r1) == np.argmin(r2) else 0

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

# ── Model loading (unchanged) ──────────────────────────────────────────────────
def initialize_model():
    print(f"Loading fine-tuned model: {FINETUNED_MODEL_NAME}")
    login(token="hf_xxxxxxxxxxx")
 
    import unsloth
    from unsloth import FastVisionModel
 
    processor = AutoProcessor.from_pretrained(FINETUNED_MODEL_NAME, use_fast=False)
 
    model, tokenizer = FastVisionModel.from_pretrained(
        FINETUNED_MODEL_NAME,
        load_in_4bit=True,
        device_map="auto",
        max_seq_length=4096,
    )
 
    model_dtype = next(
        (p.dtype for p in model.parameters() if p.dtype in (torch.float16, torch.bfloat16)),
        torch.float16
    )
    print(f"Model dtype: {model_dtype}")
 
    hidden_size = model.config.hidden_size  # NOT *2 for attention pooling
 
    # Recreate attention pooling
    attn_pooling = AttentionPooling(hidden_size).to(model.device).to(model_dtype)
    attn_pooling_path = os.path.join(FINETUNED_MODEL_NAME, "attn_pooling.pt")
    if os.path.exists(attn_pooling_path):
        attn_pooling.load_state_dict(
            torch.load(attn_pooling_path, map_location=model.device)
        )
        print(f"✓ Attention pooling loaded from {attn_pooling_path}")
    else:
        raise FileNotFoundError(f"attn_pooling.pt not found at {attn_pooling_path}")
 
    # Recreate score head — input is hidden_size (not hidden_size*2)
    score_head = nn.Sequential(
        nn.Linear(hidden_size, hidden_size // 2),
        nn.LayerNorm(hidden_size // 2),
        nn.GELU(),
        nn.Dropout(0.1),
        nn.Linear(hidden_size // 2, 1),
        nn.Sigmoid()
    ).to(model.device).to(model_dtype)
 
    score_head_path = os.path.join(FINETUNED_MODEL_NAME, "score_head.pt")
    if os.path.exists(score_head_path):
        score_head.load_state_dict(
            torch.load(score_head_path, map_location=model.device)
        )
        print(f"✓ Score head loaded from {score_head_path}")
    else:
        raise FileNotFoundError(f"score_head.pt not found at {score_head_path}")
 
    model.eval()
    score_head.eval()
    attn_pooling.eval()
 
    device = next(model.parameters()).device
    print(f"Model on device: {device}")
 
    # Return attn_pooling alongside score_head
    return model, score_head, attn_pooling, processor, device, tokenizer
 

# ── Stage 1: get scores from score head ───────────────────────────────────────
def get_scores_stage1(model, score_head, processor, device, images, inputs,
                      attn_pooling=None):
    """
    If attn_pooling is provided, use attention-weighted pooling.
    Otherwise falls back to mean+max pooling (option 1 behaviour).
    """
    model_dtype = next(
        (p.dtype for p in model.parameters() if p.dtype in (torch.float16, torch.bfloat16)),
        torch.float16
    )
 
    with torch.no_grad():
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            outputs = model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]
 
    vision_start_id  = processor.tokenizer.convert_tokens_to_ids("<|vision_start|>")
    vision_end_id    = processor.tokenizer.convert_tokens_to_ids("<|vision_end|>")
    num_images       = len(images)
    input_ids_list   = inputs['input_ids'][0].tolist()
    image_embeddings = []
 
    i = 0
    while i < len(input_ids_list) and len(image_embeddings) < num_images:
        if input_ids_list[i] == vision_start_id:
            start = i
            j = i + 1
            while j < len(input_ids_list) and input_ids_list[j] != vision_end_id:
                j += 1
            end = j + 1
            image_region = hidden_states[0, start + 1:end - 1, :]
 
            if attn_pooling is not None:
                # Step 2a: learned attention pooling
                image_region = image_region.to(attn_pooling.attention.weight.dtype)
                pooled = attn_pooling(image_region)
            else:
                # Option 1 fallback: mean + max concatenation
                mean_pool = image_region.mean(dim=0)
                max_pool  = image_region.max(dim=0).values
                pooled    = torch.cat([mean_pool, max_pool], dim=-1)
 
            image_embeddings.append(pooled)
            i = end
        else:
            i += 1
 
    if len(image_embeddings) != num_images:
        return None
 
    image_embeds = torch.stack(image_embeddings)
    image_embeds = image_embeds.to(next(score_head.parameters()).dtype)
    pred_scores  = score_head(image_embeds).squeeze(-1)
    scores       = pred_scores.detach().cpu().float().numpy().tolist()
    return scores


# ── Stage 2: generate explanation given the ranking ───────────────────────────
def generate_explanation_stage2(model, processor, tokenizer, device,
                                 sorted_images, category, group,
                                 scores, ranking):
    """
    Given the ranking from stage 1, generate a natural language explanation.
    Uses model.generate() — completely separate from score head.

    ranking: list of rank positions (0=best) for each image in order
    scores:  raw score head outputs for each image
    """
    # Build rank order string: "Image 2 > Image 1 > Image 3" etc.
    # (1-indexed for human readability)
    ranked_indices = sorted(range(len(ranking)), key=lambda x: ranking[x])
    rank_str = " > ".join([f"Image {idx+1}" for idx in ranked_indices])

    # Score descriptions per image
    score_lines = "\n".join([
        f"  Image {i+1}: persuasion score = {s:.3f}"
        for i, s in enumerate(scores)
    ])

    explanation_prompt = (
        f"You are evaluating {len(scores)} product images for the '{category}' category "
        f"(group: '{group}').\n\n"
        f"Based on visual analysis, the images have been scored and ranked as follows:\n"
        f"{score_lines}\n\n"
        f"Final ranking (best to worst): {rank_str}\n\n"
        f"For each image, provide:\n"
        f"1. A brief visual description (2-3 sentences)\n"
        f"2. What makes it more or less persuasive for selling '{category}'\n\n"
        f"Then explain the overall ranking justification in 2-3 sentences."
    )

    messages = [{
        "role": "user",
        "content": sorted_images + [{
            "type": "text",
            "text": explanation_prompt,
        }],
    }]

    prompt_text  = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=prompt_text,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        truncation=True,
        max_length=4096,
        return_tensors="pt",
    )

    model_dtype = next(
        (p.dtype for p in model.parameters() if p.dtype in (torch.float16, torch.bfloat16)),
        torch.float16
    )
    if 'pixel_values' in inputs:
        inputs['pixel_values'] = inputs['pixel_values'].to(model_dtype)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=512,      # enough for per-image descriptions + ranking justification
                do_sample=False,         # greedy — deterministic, faster
                temperature=1.0,
                repetition_penalty=1.1,  # avoid repetitive output
                pad_token_id=processor.tokenizer.eos_token_id,
            )

    # Decode only the newly generated tokens (not the prompt)
    input_length = inputs['input_ids'].shape[1]
    new_tokens   = generated_ids[0][input_length:]
    explanation  = processor.tokenizer.decode(new_tokens, skip_special_tokens=True)
    return explanation.strip()


# ── Combined inference: stage 1 + stage 2 ─────────────────────────────────────
def get_model_outputs(model, score_head, processor, tokenizer, device,
                      images, category, group, attn_pooling=None):
    """
    Stage 1: score head → ranking scores
    Stage 2: model.generate() → natural language explanation
    Returns: (scores, explanation_text)
    """
    try:
        # Prepare images (same as training)
        sorted_images = []
        for img_path in images:
            img = Image.open(img_path).convert("RGB")
            if img.width > 256 or img.height > 256:
                scale = 256 / float(max(img.width, img.height))
                img   = img.resize(
                    (int(img.width * scale), int(img.height * scale)),
                    Image.LANCZOS
                )
            sorted_images.append({"type": "image", "image": img})

        # Build stage 1 prompt (same as training prompt)
        messages = [{
            "role": "user",
            "content": sorted_images + [{
                "type": "text",
                "text": (
                    f"You are evaluating images in the '{group}' group under the '{category}' "
                    f"product category. Rank the images, based on their appeal for selling "
                    f"'{category}' product. Provide description, and *persuasion score (1-100)* "
                    f"for each image and explain the ranking."
                )
            }],
        }]

        prompt_text  = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = processor(
            text=prompt_text,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            truncation=True,
            max_length=4096,
            return_tensors="pt",
        )

        model_dtype = next(
            (p.dtype for p in model.parameters() if p.dtype in (torch.float16, torch.bfloat16)),
            torch.float16
        )
        if 'pixel_values' in inputs:
            inputs['pixel_values'] = inputs['pixel_values'].to(model_dtype)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # ── Stage 1: scores ────────────────────────────────────────────────────
        scores = get_scores_stage1(model, score_head, processor, device, images, inputs, attn_pooling=attn_pooling)

        if scores is None:
            print(f"  Warning: Could not extract image embeddings for {category}/{group}")
            return None, None

        print(f"  ✓ Stage 1 scores: {[round(s, 4) for s in scores]}")

        # ── Stage 2: explanation ───────────────────────────────────────────────
        ranking     = get_ranking_from_scores(scores, handle_ties=True)
        explanation = generate_explanation_stage2(
            model, processor, tokenizer, device,
            sorted_images, category, group, scores, ranking
        )
        print(f"  ✓ Stage 2 explanation generated ({len(explanation.split())} words)")

        return scores, explanation

    except Exception as e:
        print(f"  Error in get_model_outputs for {category}/{group}: {e}")
        import traceback
        traceback.print_exc()
        return None, None


# ── Dataset loading (unchanged) ────────────────────────────────────────────────
def load_dataset():
    data = []
    MAX_IMAGES = 4
    print("Loading dataset...")
    for category in os.listdir(dataset_image):
        category_path = os.path.join(dataset_image, category)
        if not os.path.isdir(category_path):
            continue
        for group in os.listdir(category_path):
            group_path = os.path.join(dataset_image, category, group)
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
                    "ground_truth_path": gt_path,
                })
    print(f"Dataset loaded: {len(data)} groups with ground truth")
    return data


# ── Main evaluation ────────────────────────────────────────────────────────────
def analyze_model_performance():
    all_data = load_dataset()
    train_data, test_data = train_test_split(all_data, test_size=0.15, random_state=48)
    print(f"Test set size: {len(test_data)} groups")

    # Note: tokenizer now also returned for stage 2
    model, score_head, attn_pooling, processor, device, tokenizer = initialize_model()

    def empty_stats():
        return {
            "finetuned_vs_ground_truth": [],
            "model_vs_ground_truth": defaultdict(list),
            "finetuned_vs_model": defaultdict(list),
        }

    stats = {
        "total_samples":  len(test_data),
        "valid_samples":  0,
        "agreement":        empty_stats(),
        "mse":              empty_stats(),
        "norm_ranking_loss": empty_stats(),
        "top_accuracy":     empty_stats(),
        "kendall_tau":      empty_stats(),
        "spearman_rho":     empty_stats(),
        "category_stats":   defaultdict(lambda: defaultdict(list)),
        "detailed_results": [],
    }

    details_file = os.path.join(output_dir, "detailed_results.txt")
    with open(details_file, "w") as f:
        f.write("DETAILED EVALUATION RESULTS (with explanations)\n")
        f.write("=" * 50 + "\n\n")

        for i, sample in enumerate(test_data):
            category = sample["category"]
            group    = sample["group"]
            images   = sample["images"]
            gt_path  = sample["ground_truth_path"]

            print(f"\nProcessing {i+1}/{len(test_data)}: {category}/{group}")
            f.write(f"Sample {i+1}: {category}/{group}\n")

            # Ground truth
            gt_scores, _ = extract_scores(gt_path)
            if not gt_scores:
                print("  Skipping — could not extract GT scores")
                f.write("  Skipping — could not extract GT scores\n\n")
                continue

            if len(set(gt_scores)) == 1:
                print(f"  Skipping — all GT scores tied ({gt_scores[0]})")
                f.write(f"  Skipping — all GT scores tied\n\n")
                continue

            gt_ranking = get_ranking_from_scores(gt_scores, handle_ties=True)
            f.write(f"  GT scores:  {gt_scores}\n")
            f.write(f"  GT ranking: {gt_ranking}\n")

            # Two-stage inference
            ft_scores, explanation = get_model_outputs(
                model, score_head, processor, tokenizer, device,
                images, category, group, attn_pooling = attn_pooling
            )

            if not ft_scores:
                print("  Skipping — could not get finetuned scores")
                f.write("  Skipping — could not get finetuned scores\n\n")
                continue
            if len(ft_scores) != len(gt_scores):
                print(f"  Skipping — score length mismatch")
                f.write("  Skipping — score length mismatch\n\n")
                continue

            ft_ranking = get_ranking_from_scores(ft_scores, handle_ties=True)
            f.write(f"  FT scores:  {[round(s,4) for s in ft_scores]}\n")
            f.write(f"  FT ranking: {ft_ranking}\n")

            # Save explanation to separate file
            if explanation:
                explanation_file = os.path.join(
                    explanations_dir, f"{category}_{group}_explanation.txt"
                )
                with open(explanation_file, "w", encoding="utf-8") as ef:
                    ef.write(f"Category: {category}\n")
                    ef.write(f"Group: {group}\n")
                    ef.write(f"Images: {[os.path.basename(img) for img in images]}\n")
                    ef.write(f"Score head scores: {[round(s,4) for s in ft_scores]}\n")
                    ef.write(f"Ranking (0=best): {ft_ranking}\n")
                    ef.write(f"GT scores: {gt_scores}\n")
                    ef.write(f"GT ranking: {gt_ranking}\n")
                    ef.write("\n" + "="*50 + "\n")
                    ef.write("EXPLANATION:\n")
                    ef.write("="*50 + "\n\n")
                    ef.write(explanation)
                f.write(f"  Explanation saved to: {os.path.basename(explanation_file)}\n")
                # Also write a short preview in detailed_results.txt
                preview = explanation[:200] + "..." if len(explanation) > 200 else explanation
                f.write(f"  Explanation preview: {preview}\n")

            # Metrics
            agreement = calculate_rank_agreement(gt_ranking, ft_ranking)
            mse       = calculate_mse(gt_scores, ft_scores)
            nrl       = calculate_normalized_ranking_loss(gt_ranking, ft_ranking)
            top_acc   = calculate_top_accuracy(gt_ranking, ft_ranking)
            kendall   = calculate_kendall_tau(gt_scores, ft_scores)
            spearman  = calculate_spearman_rho(gt_scores, ft_scores)

            for key, val in [("agreement", agreement), ("mse", mse),
                              ("norm_ranking_loss", nrl), ("top_accuracy", top_acc),
                              ("kendall_tau", kendall), ("spearman_rho", spearman)]:
                stats[key]["finetuned_vs_ground_truth"].append(val)
                stats["category_stats"][category][key].append(val)

            f.write(f"  FT vs GT | agreement={agreement:.4f} top_acc={top_acc} "
                    f"kendall={kendall:.4f} spearman={spearman:.4f} "
                    f"mse={mse:.4f} nrl={nrl:.4f}\n")

            # Baselines (unchanged)
            model_metrics = {}
            for model_name, file_name in MODEL_FILES.items():
                model_path = os.path.join(dataset_response, category, group, file_name)
                if not os.path.exists(model_path):
                    continue
                bl_scores, _ = extract_scores(model_path)
                if not bl_scores or len(bl_scores) != len(gt_scores):
                    continue

                bl_ranking   = get_ranking_from_scores(bl_scores, handle_ties=True)
                gt_agreement = calculate_rank_agreement(gt_ranking, bl_ranking)
                gt_mse       = calculate_mse(gt_scores, bl_scores)
                gt_nrl       = calculate_normalized_ranking_loss(gt_ranking, bl_ranking)
                gt_top_acc   = calculate_top_accuracy(gt_ranking, bl_ranking)
                gt_kendall   = calculate_kendall_tau(gt_scores, bl_scores)
                gt_spearman  = calculate_spearman_rho(gt_scores, bl_scores)

                ft_agreement = calculate_rank_agreement(ft_ranking, bl_ranking)
                ft_mse       = calculate_mse(ft_scores, bl_scores)
                ft_nrl       = calculate_normalized_ranking_loss(ft_ranking, bl_ranking)
                ft_top_acc   = calculate_top_accuracy(ft_ranking, bl_ranking)
                ft_kendall   = calculate_kendall_tau(ft_scores, bl_scores)
                ft_spearman  = calculate_spearman_rho(ft_scores, bl_scores)

                for stat_key, val in [
                    ("agreement", gt_agreement), ("mse", gt_mse),
                    ("norm_ranking_loss", gt_nrl), ("top_accuracy", gt_top_acc),
                    ("kendall_tau", gt_kendall), ("spearman_rho", gt_spearman),
                ]:
                    stats[stat_key]["model_vs_ground_truth"][model_name].append(val)

                for stat_key, val in [
                    ("agreement", ft_agreement), ("mse", ft_mse),
                    ("norm_ranking_loss", ft_nrl), ("top_accuracy", ft_top_acc),
                    ("kendall_tau", ft_kendall), ("spearman_rho", ft_spearman),
                ]:
                    stats[stat_key]["finetuned_vs_model"][model_name].append(val)

                model_metrics[model_name] = {
                    "scores": bl_scores, "ranking": bl_ranking,
                    "vs_ground_truth": {
                        "agreement": gt_agreement, "mse": gt_mse,
                        "norm_ranking_loss": gt_nrl, "top_accuracy": gt_top_acc,
                        "kendall_tau": gt_kendall, "spearman_rho": gt_spearman,
                    },
                    "vs_finetuned": {
                        "agreement": ft_agreement, "mse": ft_mse,
                        "norm_ranking_loss": ft_nrl, "top_accuracy": ft_top_acc,
                        "kendall_tau": ft_kendall, "spearman_rho": ft_spearman,
                    },
                }
                f.write(f"  {model_name} vs GT | agreement={gt_agreement:.4f} "
                        f"top_acc={gt_top_acc} kendall={gt_kendall:.4f} "
                        f"spearman={gt_spearman:.4f}\n")

            stats["valid_samples"] += 1
            stats["detailed_results"].append({
                "category": category, "group": group,
                "ground_truth_scores": gt_scores, "ground_truth_ranking": gt_ranking,
                "finetuned_scores": ft_scores,    "finetuned_ranking": ft_ranking,
                "explanation_file": f"{category}_{group}_explanation.txt",
                "metrics": {
                    "finetuned_vs_ground_truth": {
                        "agreement": agreement, "mse": mse,
                        "norm_ranking_loss": nrl, "top_accuracy": top_acc,
                        "kendall_tau": kendall, "spearman_rho": spearman,
                    },
                    "model_metrics": model_metrics,
                },
            })
            f.write("\n" + "-"*60 + "\n\n")

    # ── Aggregate (unchanged) ──────────────────────────────────────────────────
    def avg(lst):     return float(np.mean(lst)) if lst else 0.0
    def avg_inf(lst): return float(np.mean(lst)) if lst else float('inf')

    results = {
        "total_samples": stats["total_samples"],
        "valid_samples": stats["valid_samples"],
        "avg_agreement": {
            "finetuned_vs_ground_truth": avg(stats["agreement"]["finetuned_vs_ground_truth"]),
            "model_vs_ground_truth": {m: avg(v) for m, v in stats["agreement"]["model_vs_ground_truth"].items()},
        },
        "avg_mse": {
            "finetuned_vs_ground_truth": avg_inf(stats["mse"]["finetuned_vs_ground_truth"]),
            "model_vs_ground_truth": {m: avg_inf(v) for m, v in stats["mse"]["model_vs_ground_truth"].items()},
        },
        "avg_norm_ranking_loss": {
            "finetuned_vs_ground_truth": avg_inf(stats["norm_ranking_loss"]["finetuned_vs_ground_truth"]),
            "model_vs_ground_truth": {m: avg_inf(v) for m, v in stats["norm_ranking_loss"]["model_vs_ground_truth"].items()},
        },
        "avg_top_accuracy": {
            "finetuned_vs_ground_truth": avg(stats["top_accuracy"]["finetuned_vs_ground_truth"]),
            "model_vs_ground_truth": {m: avg(v) for m, v in stats["top_accuracy"]["model_vs_ground_truth"].items()},
        },
        "avg_kendall_tau": {
            "finetuned_vs_ground_truth": avg(stats["kendall_tau"]["finetuned_vs_ground_truth"]),
            "model_vs_ground_truth": {m: avg(v) for m, v in stats["kendall_tau"]["model_vs_ground_truth"].items()},
        },
        "avg_spearman_rho": {
            "finetuned_vs_ground_truth": avg(stats["spearman_rho"]["finetuned_vs_ground_truth"]),
            "model_vs_ground_truth": {m: avg(v) for m, v in stats["spearman_rho"]["model_vs_ground_truth"].items()},
        },
        "category_metrics": {
            cat: {
                "agreement":         avg(m["agreement"]),
                "mse":               avg_inf(m["mse"]),
                "norm_ranking_loss": avg_inf(m["norm_ranking_loss"]),
                "top_accuracy":      avg(m["top_accuracy"]),
                "kendall_tau":       avg(m["kendall_tau"]),
                "spearman_rho":      avg(m["spearman_rho"]),
            }
            for cat, m in stats["category_stats"].items()
        },
    }
    return results, stats


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json

    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):  return int(obj)
            if isinstance(obj, np.floating): return float(obj)
            if isinstance(obj, np.ndarray):  return obj.tolist()
            return super().default(obj)

    print("Running evaluation with two-stage inference (scores + explanations)...")
    results, raw_stats = analyze_model_performance()

    with open(os.path.join(output_dir, "comprehensive_evaluation_results.json"), "w") as f:
        json.dump(results, f, indent=2, cls=NpEncoder)
    with open(os.path.join(output_dir, "detailed_stats.json"), "w") as f:
        json.dump({"detailed_results": raw_stats["detailed_results"]}, f, indent=2, cls=NpEncoder)

    print("\n===== EVALUATION SUMMARY =====")
    print(f"Total samples: {results['total_samples']}  |  Valid: {results['valid_samples']}")

    all_models = list(results['avg_top_accuracy']['model_vs_ground_truth'].keys())

    for metric_name, metric_key, higher_better in [
        ("Top Accuracy",      "avg_top_accuracy",      True),
        ("Agreement",         "avg_agreement",          True),
        ("Kendall's Tau",     "avg_kendall_tau",        True),
        ("Spearman's Rho",    "avg_spearman_rho",       True),
        ("Norm Ranking Loss", "avg_norm_ranking_loss",  False),
        ("MSE",               "avg_mse",                False),
    ]:
        print(f"\n----- {metric_name} ({'↑' if higher_better else '↓'}) -----")
        ft_val = results[metric_key]["finetuned_vs_ground_truth"]
        print(f"  fine-tuned : {ft_val:.4f}")
        ranking_list = [("fine-tuned", ft_val)]
        for m in all_models:
            v = results[metric_key]["model_vs_ground_truth"][m]
            print(f"  {m:20s}: {v:.4f}")
            ranking_list.append((m, v))
        ranking_list.sort(key=lambda x: x[1], reverse=higher_better)
        best = ranking_list[0]
        if best[0] == "fine-tuned":
            print(f"  → Fine-tuned is BEST")
        else:
            diff = abs(ft_val - best[1])
            print(f"  → Best baseline: {best[0]} ({best[1]:.4f}), gap = {diff:.4f}")

    print("\n===== CATEGORY-WISE (fine-tuned) =====")
    for cat, m in results["category_metrics"].items():
        print(f"  {cat}: top_acc={m['top_accuracy']:.3f} "
              f"kendall={m['kendall_tau']:.3f} "
              f"spearman={m['spearman_rho']:.3f} "
              f"agreement={m['agreement']:.3f}")

    print(f"\nResults saved to {output_dir}")
    print(f"Explanations saved to {explanations_dir}/ ({results['valid_samples']} files)")