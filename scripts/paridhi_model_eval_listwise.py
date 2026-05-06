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


# Import your existing function
from simple_data_preprocess import extract_text_data
import random 
import torch 
import numpy as np
SEED = 50
random.seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


## SET PATHS
data_root = "home/debajyoti/paridhi_mtp/product_images_real"
#data_root = "/home/deepg/NAS/Downloads/MTP-2-persuasion/dataset"
dataset_image = os.path.join(data_root, "dataset_image_new")
dataset_response = os.path.join(data_root, "dataset_response_new")
# Updated path to user preferred outputs (ground truth)
dataset_user_preferred = os.path.join(data_root, "final_data")
output_dir = "./listwise_model_evaluation"
os.makedirs(output_dir, exist_ok=True)

# Model mappings
MODEL_FILES = {
    "qwen_zeroshot": "output_qwen_zeroshot.txt",
    "qwen_fewshot": "output_qwen_fewshot.txt",
    "pixtral_zeroshot": "output_pixtral_zeroshot.txt",
    "pixtral_fewshot": "output_pixtral_fewshot.txt"
}

# Fine-tuned model information
#FINETUNED_MODEL_NAME = "Deb123/qwen2.5-vl-7b-pair-tournament-final-finetuned-private"
#FINETUNED_MODEL_NAME = "vlm_finetuned_full_context_48_2"
FINETUNED_MODEL_NAME = "vlm_finetuned_listwise"
#FINETUNED_MODEL_NAME = "Deb123/qwen2.5-vl-7b-pair-finetuned-private"
def calculate_kendall_tau(ranking1, ranking2):
    """
    Kendall's Tau rank correlation
    Returns value in [-1, 1]
    """
    if not ranking1 or not ranking2 or len(ranking1) != len(ranking2):
        return 0.0
    tau, _ = kendalltau(ranking1, ranking2)
    return 0.0 if np.isnan(tau) else float(tau)

def calculate_spearman_rho(ranking1, ranking2):
    """
    Spearman's Rho rank correlation
    Returns value in [-1, 1]
    """
    if not ranking1 or not ranking2 or len(ranking1) != len(ranking2):
        return 0.0
    rho, _ = spearmanr(ranking1, ranking2)
    return 0.0 if np.isnan(rho) else float(rho)

def get_ranking_from_scores(scores, handle_ties=True):
    """
    Convert scores to rankings (higher score = better rank)
    Args:
        scores: List of scores
        handle_ties: If True, assign the same rank to tied scores
    """
    if not scores:
        return []
    
    # Create (index, score) pairs and sort by score (descending)
    pairs = [(i, score) for i, score in enumerate(scores)]
    sorted_pairs = sorted(pairs, key=lambda x: x[1], reverse=True)
    
    if handle_ties:
        # Handle ties (same scores get same rank)
        ranking = [0] * len(scores)
        current_rank = 0
        last_score = None
        
        for i, (idx, score) in enumerate(sorted_pairs):
            if i > 0 and score == last_score:
                # This is a tie, use the same rank as previous
                ranking[idx] = current_rank
            else:
                # New rank
                current_rank = i
                ranking[idx] = current_rank
            last_score = score
    else:
        # No tie handling (original method)
        ranking = [0] * len(scores)
        for rank, (idx, _) in enumerate(sorted_pairs):
            ranking[idx] = rank
    
    return ranking

def calculate_rank_agreement(ranking1, ranking2):
    """
    Calculate the agreement between two rankings
    Returns the percentage of positions where the rankings agree
    """
    if not ranking1 or not ranking2 or len(ranking1) != len(ranking2):
        return 0
    
    agreement = sum(r1 == r2 for r1, r2 in zip(ranking1, ranking2))
    return agreement / len(ranking1)

def calculate_mse(scores1, scores2):
    """
    Calculate the Mean Squared Error between two sets of scores
    """
    if not scores1 or not scores2 or len(scores1) != len(scores2):
        return float('inf')
    
    return np.mean((np.array(scores1) - np.array(scores2)) ** 2)

def calculate_normalized_ranking_loss(ranking1, ranking2):
    """
    Calculate the normalized ranking loss between two rankings
    This measures the euclidean distance between rankings, normalized by the length
    """
    if not ranking1 or not ranking2 or len(ranking1) != len(ranking2):
        return float('inf')
    
    # Convert to numpy arrays for vector operations
    r1 = np.array(ranking1)
    r2 = np.array(ranking2)
    
    # Calculate normalized distance
    distance = np.linalg.norm(r1 - r2) / len(r1)
    return distance

def calculate_top_accuracy(ranking1, ranking2):
    """
    Calculate the accuracy of top-ranked item prediction
    Returns 1 if the top-ranked items match, 0 otherwise
    """
    if not ranking1 or not ranking2 or len(ranking1) != len(ranking2):
        return 0
    
    # Find indices of top-ranked items (smallest rank number)
    top1 = np.argmin(ranking1)
    top2 = np.argmin(ranking2)
    
    return 1 if top1 == top2 else 0

def extract_scores(file_path):
    """Extract scores from a file using extract_text_data and handle errors"""
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

def initialize_model():
    """Load the fine-tuned model and processor"""
    print(f"Loading fine-tuned model: {FINETUNED_MODEL_NAME}")
    
    # Authenticate
    login(token="hf_xxxxxxxxxxxxxxxxxx")
    
    # Load model and processor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(FINETUNED_MODEL_NAME, use_fast=False)
    
    # Load the base model
    from transformers import AutoModelForImageTextToText
    model = AutoModelForImageTextToText.from_pretrained(FINETUNED_MODEL_NAME).to(device)
    
    # Recreate score head architecture (MUST match training exactly!)
    hidden_size = model.config.hidden_size * 2
    model.score_head = nn.Sequential(
        nn.Linear(hidden_size, hidden_size // 2),
        nn.LayerNorm(hidden_size // 2),
        nn.GELU(),
        nn.Dropout(0.1),
        nn.Linear(hidden_size // 2, 1)
        #nn.Sigmoid()
    ).to(device)
    
    # Load saved score head weights
    score_head_path = os.path.join(FINETUNED_MODEL_NAME, "score_head.pt")
    if os.path.exists(score_head_path):
        model.score_head.load_state_dict(torch.load(score_head_path, map_location=device))
        print(f"✓ Score head loaded from {score_head_path}")
    else:
        raise FileNotFoundError(
            f"Score head not found at {score_head_path}!\n"
            f"You need to retrain the model with the updated saving code."
        )
    
    # Convert score head to same dtype as model (float16)
    model.score_head = model.score_head.half()
    
    # Set to eval mode
    model.eval()
    model.score_head.eval()
    
    print(f"Model and score head loaded successfully on {device}!")
    print(f"Model dtype: {next(model.parameters()).dtype}")
    print(f"Score head dtype: {next(model.score_head.parameters()).dtype}")
    
    return model, processor, device

def get_image_embeddings_from_vision_tokens(hidden_states, input_ids, processor, num_images):
    """
    Extract image embeddings by finding vision token regions and pooling them
    """
    # vision_start_id = 151652  # <|vision_start|>
    # vision_end_id = 151653    # <|vision_end|>

    vision_start_id = processor.tokenizer.convert_tokens_to_ids("<|vision_start|>")
    vision_end_id = processor.tokenizer.convert_tokens_to_ids("<|vision_end|>")
    
    input_ids_list = input_ids[0].tolist()
    image_embeddings = []
    
    i = 0
    while i < len(input_ids_list) and len(image_embeddings) < num_images:
        if input_ids_list[i] == vision_start_id:
            # Find the end of this vision region
            start = i
            j = i + 1
            while j < len(input_ids_list) and input_ids_list[j] != vision_end_id:
                j += 1
            end = j + 1  # Include vision_end token
            
            # Extract and pool embeddings for this image
            image_region_embeds = hidden_states[0, start+1:end-1, :]
            # pooled_embed = image_region_embeds.mean(dim=0)  # Mean pooling
            # mean + max pool
            mean_pool = image_region_embeds.mean(dim=0)
            max_pool = image_region_embeds.max(dim=0).values

            pooled_embed = torch.cat([mean_pool, max_pool], dim=-1)
            image_embeddings.append(pooled_embed)
            
            i = end
        else:
            i += 1
    
    if len(image_embeddings) != num_images:
        print(f"Warning: Found {len(image_embeddings)} vision regions, expected {num_images}")
        # Fallback: just return None
        return None
    
    return torch.stack(image_embeddings)

def get_model_outputs(model, processor, device, images, category, group):
    """
    Get scores directly from the score head instead of parsing generated text
    This is more reliable than text generation
    """
    try:
        # Prepare images for processing
        sorted_images = []
        for img_path in images:
            img = Image.open(img_path).convert("RGB")
            # Resize if needed (same as training)
            if img.width > 256 or img.height > 256:
                scaling_factor = 256 / float(max(img.width, img.height))
                new_width = int(img.width * scaling_factor)
                new_height = int(img.height * scaling_factor)
                img = img.resize((new_width, new_height), Image.LANCZOS)
            sorted_images.append({"type": "image", "image": img})
        
        # Create the same prompt as training
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
        
        # Process inputs (same as training)
        from qwen_vl_utils import process_vision_info
        
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
        
        # Forward pass to get hidden states
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]
        
        # Extract image embeddings and get scores from score head
        num_images = len(images)

        # CRITICAL FIX: Extract image embeddings from vision token regions
        image_embeds = get_image_embeddings_from_vision_tokens(
            hidden_states, 
            inputs['input_ids'], 
            processor,
            num_images
        )
        
        if image_embeds is None:
            print(f"Failed to extract image embeddings for {category}/{group}")
            return None, None
        
        # Get scores from score head (ou
        pred_scores = model.score_head(image_embeds).squeeze(-1)
        
        # Convert to numpy and scale to 0-100 range for comparison with ground truth
        # scores = pred_scores.detach().cpu().numpy()
        # scores_scaled = (scores * 100).tolist()  # Scale from [0,1] to [0,100]

        scores = pred_scores.detach().cpu().float().numpy()
        scores_scaled = scores.tolist()
        
        print(f"✓ Successfully extracted scores from score head: {scores_scaled}")
        
        # Create a simple text output for logging (optional)
        output_text = f"Predicted scores (from score head): {scores_scaled}\n"
        output_text += f"Normalized scores: {scores.tolist()}\n"
        
        # Save to file for debugging
        output_file = os.path.join(output_dir, f"{category}_{group}_finetuned_scores.txt")
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(output_text)

        # In get_model_outputs, add this debug code:
        print(f"\n=== DEBUG: Hidden State Analysis ===")
        print(f"Input IDs shape: {inputs['input_ids'].shape}")
        print(f"Hidden states shape: {hidden_states.shape}")
        print(f"First 20 token IDs: {inputs['input_ids'][0, :20].tolist()}")

        # Check what tokens are at the positions you're extracting
        extracted_token_ids = inputs['input_ids'][0, :num_images].tolist()
        extracted_tokens = processor.tokenizer.convert_ids_to_tokens(extracted_token_ids)
        print(f"Tokens at positions [0:{num_images}]: {extracted_tokens}")

        # Get special token IDs
        special_tokens = {
            'vision_start': processor.tokenizer.convert_tokens_to_ids("<|vision_start|>"),
            'vision_end': processor.tokenizer.convert_tokens_to_ids("<|vision_end|>"),
            'image_pad': processor.tokenizer.convert_tokens_to_ids("<|image_pad|>"),
        }
        print(f"Special token IDs: {special_tokens}")
        print(f"================================\n")
        
        return scores_scaled, output_text
        
    except Exception as e:
        print(f"Error in get_model_outputs for {category}/{group}: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def load_dataset():
    """Load all data from the dataset matching the original approach"""
    data = []
    MAX_IMAGES = 4  # From your code

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
                    
                    # Check that ground truth file exists without extracting yet
                    user_preferred_path = os.path.join(dataset_user_preferred, category, group, "user_output.txt")
                    
                    if os.path.exists(user_preferred_path):
                        # Include this group without extracting scores yet
                        data.append({
                            "images": images,
                            "category": category,
                            "group": group,
                            "ground_truth_path": user_preferred_path
                        })

    print(f"Dataset loaded with {len(data)} groups having ground truth")
    return data

def analyze_model_performance():
    """Comprehensive evaluation of fine-tuned model against ground truth and baseline models"""
    # Load the dataset with ground truth
    all_data = load_dataset()
    
    # Split the dataset (use the same random_state as in your training)
    train_data, test_data = train_test_split(all_data, test_size=0.05, random_state=48)
    
    print(f"Validation set size: {len(test_data)} groups")
    
    # Initialize the fine-tuned model
    model, processor, device = initialize_model()
    
    # Initialize stats
    stats = {
        "total_samples": len(test_data),
        "valid_samples": 0,
        
        # Agreement metrics
        "agreement": {
            "finetuned_vs_ground_truth": [],
            "model_vs_ground_truth": defaultdict(list),
            "finetuned_vs_model": defaultdict(list),
        },
        
        # MSE metrics
        "mse": {
            "finetuned_vs_ground_truth": [],
            "model_vs_ground_truth": defaultdict(list),
            "finetuned_vs_model": defaultdict(list),
        },
        
        # Normalized ranking loss metrics
        "norm_ranking_loss": {
            "finetuned_vs_ground_truth": [],
            "model_vs_ground_truth": defaultdict(list),
            "finetuned_vs_model": defaultdict(list),
        },
        
        # Top accuracy metrics
        "top_accuracy": {
            "finetuned_vs_ground_truth": [],
            "model_vs_ground_truth": defaultdict(list),
            "finetuned_vs_model": defaultdict(list),
        },

        # Rank correlation metrics (LISTWISE-ALIGNED)
        "kendall_tau": {
            "finetuned_vs_ground_truth": [],
            "model_vs_ground_truth": defaultdict(list),
            "finetuned_vs_model": defaultdict(list),
        },
        "spearman_rho": {
            "finetuned_vs_ground_truth": [],
            "model_vs_ground_truth": defaultdict(list),
            "finetuned_vs_model": defaultdict(list),
        },
        
        # Additional stats
        "category_stats": defaultdict(lambda: defaultdict(list)),
        "detailed_results": []

        

    }
    
    # Open file for detailed results
    details_file = os.path.join(output_dir, "detailed_results.txt")
    with open(details_file, "w", encoding="utf-8") as detailed_f:
        detailed_f.write("DETAILED EVALUATION RESULTS\n")
        detailed_f.write("=========================\n\n")
        
        # Process each sample in validation set
        for i, sample in enumerate(test_data):
            category = sample["category"]
            group = sample["group"]
            images = sample["images"]
            ground_truth_path = sample["ground_truth_path"]
            
            print(f"Processing {i+1}/{len(test_data)}: {category}/{group}")
            detailed_f.write(f"Sample {i+1}: {category}/{group}\n")
            
            # Extract ground truth scores from file path
            ground_truth_scores, ground_truth_info = extract_scores(ground_truth_path)
            print("\nDEBUG --- Ground Truth")
            print("GT scores:", ground_truth_scores)
            print("GT ranking:", get_ranking_from_scores(ground_truth_scores))
            
            if not ground_truth_scores:
                print(f"Could not extract ground truth scores")
                detailed_f.write(f"Could not extract ground truth scores\n\n")
                continue
            
            # Compute ground truth ranking
            ground_truth_ranking = get_ranking_from_scores(ground_truth_scores, handle_ties=True)
            detailed_f.write(f"  Ground truth scores: {ground_truth_scores}\n")
            #filter out all-tied ground truth samples
            if len(set(ground_truth_scores)) == 1:
                print(f"Skipping {category}/{group} - all GT scores are tied ({ground_truth_scores[0]})")
                detailed_f.write(f"Skipping - all GT scores tied\n\n")
                continue
            detailed_f.write(f"  Ground truth ranking: {ground_truth_ranking}\n")
            
            # Get fine-tuned model scores by running inference
            print(f"  Generating finetuned model output...")
            finetuned_scores, finetuned_output = get_model_outputs(model, processor, device, images, category, group)
            print("\nDEBUG --- Model Prediction")
            print("Pred scores:", finetuned_scores)
            print("Pred ranking:", get_ranking_from_scores(finetuned_scores))
            
            if not finetuned_scores:
                print(f"Could not generate scores from fine-tuned model")
                detailed_f.write(f"Could not generate scores from fine-tuned model\n\n")
                continue
                
            detailed_f.write(f"  Fine-tuned model scores: {finetuned_scores}\n")
            
            # Ensure score lengths match
            if len(finetuned_scores) != len(ground_truth_scores):
                print(f"Score length mismatch: finetuned={len(finetuned_scores)}, ground_truth={len(ground_truth_scores)}")
                detailed_f.write(f"Score length mismatch\n\n")
                continue
            
            # Get finetuned ranking
            finetuned_ranking = get_ranking_from_scores(finetuned_scores, handle_ties=True)
            detailed_f.write(f"  Fine-tuned model ranking: {finetuned_ranking}\n")
            
            # Calculate metrics between fine-tuned model and ground truth
            agreement = calculate_rank_agreement(ground_truth_ranking, finetuned_ranking)
            mse = calculate_mse(ground_truth_scores, finetuned_scores)
            norm_rank_loss = calculate_normalized_ranking_loss(ground_truth_ranking, finetuned_ranking)
            top_acc = calculate_top_accuracy(ground_truth_ranking, finetuned_ranking)

            # kendall = calculate_kendall_tau(ground_truth_ranking, finetuned_ranking)
            # spearman = calculate_spearman_rho(ground_truth_ranking, finetuned_ranking)

            kendall = calculate_kendall_tau(ground_truth_scores, finetuned_scores)
            spearman = calculate_spearman_rho(ground_truth_scores, finetuned_scores)

            stats["kendall_tau"]["finetuned_vs_ground_truth"].append(kendall)
            stats["spearman_rho"]["finetuned_vs_ground_truth"].append(spearman)

            
            # Add to global stats
            stats["agreement"]["finetuned_vs_ground_truth"].append(agreement)
            stats["mse"]["finetuned_vs_ground_truth"].append(mse)
            stats["norm_ranking_loss"]["finetuned_vs_ground_truth"].append(norm_rank_loss)
            stats["top_accuracy"]["finetuned_vs_ground_truth"].append(top_acc)
            
            # Add to category stats
            stats["category_stats"][category]["agreement"].append(agreement)
            stats["category_stats"][category]["mse"].append(mse)
            stats["category_stats"][category]["norm_ranking_loss"].append(norm_rank_loss)
            stats["category_stats"][category]["top_accuracy"].append(top_acc)
            
            detailed_f.write(f"  === Metrics: Fine-tuned vs Ground Truth ===\n")
            detailed_f.write(f"  Agreement: {agreement:.4f}\n")
            detailed_f.write(f"  MSE: {mse:.4f}\n")
            detailed_f.write(f"  Normalized Ranking Loss: {norm_rank_loss:.4f}\n")
            detailed_f.write(f"  Top Accuracy: {top_acc}\n\n")
            detailed_f.write(f"  Kendall's Tau: {kendall:.4f}\n")
            detailed_f.write(f"  Spearman's Rho: {spearman:.4f}\n")

            
            # Compare with each baseline model
            #############################lenient version - paridhi################################
            # Compare with each baseline model (OPTIONAL - don't affect valid_sample count)
            model_metrics = {}
            
            for model_name, file_name in MODEL_FILES.items():
                model_path = os.path.join(dataset_response, category, group, file_name)
                
                # Skip if file doesn't exist (don't fail the whole sample)
                if not os.path.exists(model_path):
                    print(f"  Skipping {model_name} - file not found")
                    detailed_f.write(f"  Skipping {model_name} - file not found\n")
                    continue
                
                model_scores, _ = extract_scores(model_path)
                
                # Skip if extraction failed (don't fail the whole sample)
                if not model_scores:
                    print(f"  Skipping {model_name} - extraction failed (possibly ERROR in file)")
                    detailed_f.write(f"  Skipping {model_name} - extraction failed\n")
                    continue
                # if model_scores == ground_truth_scores:
                #     print(f"  Skipping {model_name} - extracted scores match ground truth (invalid baseline)")
                #     detailed_f.write(f"  Skipping {model_name} - extracted scores match ground truth\n")
                #     continue
                
                # Skip if score length doesn't match (don't fail the whole sample)
                if len(model_scores) != len(ground_truth_scores):
                    print(f"  Skipping {model_name} - score length mismatch: {len(model_scores)} vs {len(ground_truth_scores)}")
                    detailed_f.write(f"  Skipping {model_name} - score length mismatch\n")
                    continue
                
                # If we reach here, this baseline is valid - process it
                print(f"  ✓ Processing {model_name}")
                model_ranking = get_ranking_from_scores(model_scores, handle_ties=True)
                
                # Calculate metrics with ground truth
                gt_agreement = calculate_rank_agreement(ground_truth_ranking, model_ranking)
                gt_mse = calculate_mse(ground_truth_scores, model_scores)
                gt_norm_rank_loss = calculate_normalized_ranking_loss(ground_truth_ranking, model_ranking)
                gt_top_acc = calculate_top_accuracy(ground_truth_ranking, model_ranking)

                # gt_kendall = calculate_kendall_tau(ground_truth_ranking, model_ranking)
                # gt_spearman = calculate_spearman_rho(ground_truth_ranking, model_ranking)

                # ft_kendall = calculate_kendall_tau(finetuned_ranking, model_ranking)
                # ft_spearman = calculate_spearman_rho(finetuned_ranking, model_ranking)

                gt_kendall = calculate_kendall_tau(ground_truth_scores, model_scores)
                gt_spearman = calculate_spearman_rho(ground_truth_scores, model_scores)

                ft_kendall = calculate_kendall_tau(finetuned_scores, model_scores)
                ft_spearman = calculate_spearman_rho(finetuned_scores, model_scores)
                
                # Calculate metrics with fine-tuned model
                ft_agreement = calculate_rank_agreement(finetuned_ranking, model_ranking)
                ft_mse = calculate_mse(finetuned_scores, model_scores)
                ft_norm_rank_loss = calculate_normalized_ranking_loss(finetuned_ranking, model_ranking)
                ft_top_acc = calculate_top_accuracy(finetuned_ranking, model_ranking)
                
                # Add to global stats
                stats["agreement"]["model_vs_ground_truth"][model_name].append(gt_agreement)
                stats["mse"]["model_vs_ground_truth"][model_name].append(gt_mse)
                stats["norm_ranking_loss"]["model_vs_ground_truth"][model_name].append(gt_norm_rank_loss)
                stats["top_accuracy"]["model_vs_ground_truth"][model_name].append(gt_top_acc)
                
                stats["agreement"]["finetuned_vs_model"][model_name].append(ft_agreement)
                stats["mse"]["finetuned_vs_model"][model_name].append(ft_mse)
                stats["norm_ranking_loss"]["finetuned_vs_model"][model_name].append(ft_norm_rank_loss)
                stats["top_accuracy"]["finetuned_vs_model"][model_name].append(ft_top_acc)

                stats["kendall_tau"]["model_vs_ground_truth"][model_name].append(gt_kendall)
                stats["spearman_rho"]["model_vs_ground_truth"][model_name].append(gt_spearman)

                stats["kendall_tau"]["finetuned_vs_model"][model_name].append(ft_kendall)
                stats["spearman_rho"]["finetuned_vs_model"][model_name].append(ft_spearman)

                
                model_metrics[model_name] = {
                    "scores": model_scores,
                    "ranking": model_ranking,
                    "vs_ground_truth": {
                        "agreement": gt_agreement,
                        "mse": gt_mse,
                        "norm_ranking_loss": gt_norm_rank_loss,
                        "top_accuracy": gt_top_acc,
                        "kendall_tau": gt_kendall,
                        "spearman_rho": gt_spearman,
                    },
                    "vs_finetuned": {
                        "agreement": ft_agreement,
                        "mse": ft_mse,
                        "norm_ranking_loss": ft_norm_rank_loss,
                        "top_accuracy": ft_top_acc,
                        "kendall_tau": ft_kendall,
                        "spearman_rho": ft_spearman,
                    }
                }
                
                detailed_f.write(f"  === {model_name} Metrics ===\n")
                detailed_f.write(f"  {model_name} scores: {model_scores}\n")
                detailed_f.write(f"  {model_name} ranking: {model_ranking}\n")
                detailed_f.write(f"  vs Ground Truth:\n")
                detailed_f.write(f"    Agreement: {gt_agreement:.4f}\n")
                detailed_f.write(f"    MSE: {gt_mse:.4f}\n")
                detailed_f.write(f"    Normalized Ranking Loss: {gt_norm_rank_loss:.4f}\n")
                detailed_f.write(f"    Top Accuracy: {gt_top_acc}\n")
                detailed_f.write(f"  vs Fine-tuned Model:\n")
                detailed_f.write(f"    Agreement: {ft_agreement:.4f}\n")
                detailed_f.write(f"    MSE: {ft_mse:.4f}\n")
                detailed_f.write(f"    Normalized Ranking Loss: {ft_norm_rank_loss:.4f}\n")
                detailed_f.write(f"    Top Accuracy: {ft_top_acc}\n\n")
                detailed_f.write(f"    Kendall's Tau: {gt_kendall:.4f}\n")
                detailed_f.write(f"    Spearman's Rho: {gt_spearman:.4f}\n")

            
            # ALWAYS count this sample as valid if fine-tuned model worked
            stats["valid_samples"] += 1
            
            # Store detailed result (even if not all baselines are available)
            stats["detailed_results"].append({
                "category": category,
                "group": group,
                "ground_truth_scores": ground_truth_scores,
                "ground_truth_ranking": ground_truth_ranking,
                "finetuned_scores": finetuned_scores,
                "finetuned_ranking": finetuned_ranking,
                "metrics": {
                    "finetuned_vs_ground_truth": {
                        "agreement": agreement,
                        "mse": mse,
                        "norm_ranking_loss": norm_rank_loss,
                        "top_accuracy": top_acc,
                    },
                    "model_metrics": model_metrics  # Will only contain available baselines
                }
            })
            
            detailed_f.write("\n" + "-" * 60 + "\n\n")
            
    # Calculate average metrics
    results = {
        "total_samples": stats["total_samples"],
        "valid_samples": stats["valid_samples"],
        
        # Calculate averages for agreement
        "avg_agreement": {
            "finetuned_vs_ground_truth": float(np.mean(stats["agreement"]["finetuned_vs_ground_truth"])) if stats["agreement"]["finetuned_vs_ground_truth"] else 0,
            "model_vs_ground_truth": {model: float(np.mean(agreements)) if agreements else 0 
                                     for model, agreements in stats["agreement"]["model_vs_ground_truth"].items()},
            "finetuned_vs_model": {model: float(np.mean(agreements)) if agreements else 0 
                                  for model, agreements in stats["agreement"]["finetuned_vs_model"].items()},
        },
        
        # Calculate averages for MSE
        "avg_mse": {
            "finetuned_vs_ground_truth": float(np.mean(stats["mse"]["finetuned_vs_ground_truth"])) if stats["mse"]["finetuned_vs_ground_truth"] else float('inf'),
            "model_vs_ground_truth": {model: float(np.mean(mses)) if mses else float('inf') 
                                     for model, mses in stats["mse"]["model_vs_ground_truth"].items()},
            "finetuned_vs_model": {model: float(np.mean(mses)) if mses else float('inf') 
                                  for model, mses in stats["mse"]["finetuned_vs_model"].items()},
        },
        
        # Calculate averages for normalized ranking loss
        "avg_norm_ranking_loss": {
            "finetuned_vs_ground_truth": float(np.mean(stats["norm_ranking_loss"]["finetuned_vs_ground_truth"])) if stats["norm_ranking_loss"]["finetuned_vs_ground_truth"] else float('inf'),
            "model_vs_ground_truth": {model: float(np.mean(losses)) if losses else float('inf') 
                                     for model, losses in stats["norm_ranking_loss"]["model_vs_ground_truth"].items()},
            "finetuned_vs_model": {model: float(np.mean(losses)) if losses else float('inf') 
                                  for model, losses in stats["norm_ranking_loss"]["finetuned_vs_model"].items()},
        },
        
        # Calculate averages for top accuracy
        "avg_top_accuracy": {
            "finetuned_vs_ground_truth": float(np.mean(stats["top_accuracy"]["finetuned_vs_ground_truth"])) if stats["top_accuracy"]["finetuned_vs_ground_truth"] else 0,
            "model_vs_ground_truth": {model: float(np.mean(accs)) if accs else 0 
                                     for model, accs in stats["top_accuracy"]["model_vs_ground_truth"].items()},
            "finetuned_vs_model": {model: float(np.mean(accs)) if accs else 0 
                                  for model, accs in stats["top_accuracy"]["finetuned_vs_model"].items()},
        },

        "avg_kendall_tau": {
            "finetuned_vs_ground_truth": float(np.mean(stats["kendall_tau"]["finetuned_vs_ground_truth"])) 
                if stats["kendall_tau"]["finetuned_vs_ground_truth"] else 0,
            "model_vs_ground_truth": {
                model: float(np.mean(vals)) for model, vals in stats["kendall_tau"]["model_vs_ground_truth"].items()
            },
        },
        "avg_spearman_rho": {
            "finetuned_vs_ground_truth": float(np.mean(stats["spearman_rho"]["finetuned_vs_ground_truth"])) 
                if stats["spearman_rho"]["finetuned_vs_ground_truth"] else 0,
            "model_vs_ground_truth": {
                model: float(np.mean(vals)) for model, vals in stats["spearman_rho"]["model_vs_ground_truth"].items()
            },
        },

        
        # Calculate category-wise metrics
        "category_metrics": {}
    }
    
    # Process category stats
    for category, metrics in stats["category_stats"].items():
        results["category_metrics"][category] = {
            "agreement": float(np.mean(metrics["agreement"])) if metrics["agreement"] else 0,
            "mse": float(np.mean(metrics["mse"])) if metrics["mse"] else float('inf'),
            "norm_ranking_loss": float(np.mean(metrics["norm_ranking_loss"])) if metrics["norm_ranking_loss"] else float('inf'),
            "top_accuracy": float(np.mean(metrics["top_accuracy"])) if metrics["top_accuracy"] else 0,
        }
    
    return results, stats

if __name__ == "__main__":
    print("Running comprehensive model evaluation...")
    results, raw_stats = analyze_model_performance()
    
    # Save results to JSON
    import json
    
    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super(NpEncoder, self).default(obj)
    
    with open(os.path.join(output_dir, "comprehensive_evaluation_results.json"), "w") as f:
        json.dump(results, f, indent=2, cls=NpEncoder)
    
    # Also save detailed stats for further analysis
    with open(os.path.join(output_dir, "detailed_stats.json"), "w") as f:
        json.dump({"detailed_results": raw_stats["detailed_results"]}, f, indent=2, cls=NpEncoder)
    
    # Print summary of results
    print("\n===== COMPREHENSIVE MODEL EVALUATION SUMMARY =====")
    print(f"Total validation samples: {results['total_samples']}")
    print(f"Valid samples with complete data: {results['valid_samples']}")
    
    print("\n----- Agreement Metrics -----")
    print(f"Fine-tuned model vs Ground Truth: {results['avg_agreement']['finetuned_vs_ground_truth']:.4f}")
    
    all_models = list(results['avg_agreement']['model_vs_ground_truth'].keys())
    for model in all_models:
        print(f"{model} vs Ground Truth: {results['avg_agreement']['model_vs_ground_truth'][model]:.4f}")
    
    print("\n----- MSE Metrics -----")
    print(f"Fine-tuned model vs Ground Truth: {results['avg_mse']['finetuned_vs_ground_truth']:.4f}")
    
    for model in all_models:
        print(f"{model} vs Ground Truth: {results['avg_mse']['model_vs_ground_truth'][model]:.4f}")
    
    print("\n----- Normalized Ranking Loss Metrics -----")
    print(f"Fine-tuned model vs Ground Truth: {results['avg_norm_ranking_loss']['finetuned_vs_ground_truth']:.4f}")
    
    for model in all_models:
        print(f"{model} vs Ground Truth: {results['avg_norm_ranking_loss']['model_vs_ground_truth'][model]:.4f}")
    
    print("\n----- Top Accuracy Metrics -----")
    print(f"Fine-tuned model vs Ground Truth: {results['avg_top_accuracy']['finetuned_vs_ground_truth']:.4f}")
    
    for model in all_models:
        print(f"{model} vs Ground Truth: {results['avg_top_accuracy']['model_vs_ground_truth'][model]:.4f}")

    print("\n----- Rank Correlation Metrics (Listwise) -----")
    print(f"Fine-tuned vs Ground Truth | Kendall's Tau: {results['avg_kendall_tau']['finetuned_vs_ground_truth']:.4f}")
    print(f"Fine-tuned vs Ground Truth | Spearman's Rho: {results['avg_spearman_rho']['finetuned_vs_ground_truth']:.4f}")

    for model in all_models:
        print(f"{model} | Kendall's Tau: {results['avg_kendall_tau']['model_vs_ground_truth'][model]:.4f}")
        print(f"{model} | Spearman's Rho: {results['avg_spearman_rho']['model_vs_ground_truth'][model]:.4f}")

    
    # Find best model for each metric
    print("\n===== MODEL RANKINGS BY METRIC =====")
    
    # Agreement ranking (higher is better)
    print("\nRanking by Agreement (higher is better):")
    agreement_ranking = [("fine-tuned", results['avg_agreement']['finetuned_vs_ground_truth'])]
    for model, score in results['avg_agreement']['model_vs_ground_truth'].items():
        agreement_ranking.append((model, score))
    agreement_ranking.sort(key=lambda x: x[1], reverse=True)
    
    for i, (model, score) in enumerate(agreement_ranking, 1):
        print(f"{i}. {model}: {score:.4f}")
    
    # MSE ranking (lower is better)
    print("\nRanking by MSE (lower is better):")
    mse_ranking = [("fine-tuned", results['avg_mse']['finetuned_vs_ground_truth'])]
    for model, score in results['avg_mse']['model_vs_ground_truth'].items():
        mse_ranking.append((model, score))
    mse_ranking.sort(key=lambda x: x[1])
    
    for i, (model, score) in enumerate(mse_ranking, 1):
        print(f"{i}. {model}: {score:.4f}")
    
    ## Normalized Ranking Loss ranking (lower is better)
    print("\nRanking by Normalized Ranking Loss (lower is better):")
    nrl_ranking = [("fine-tuned", results['avg_norm_ranking_loss']['finetuned_vs_ground_truth'])]
    for model, score in results['avg_norm_ranking_loss']['model_vs_ground_truth'].items():
        nrl_ranking.append((model, score))
    nrl_ranking.sort(key=lambda x: x[1])
    
    for i, (model, score) in enumerate(nrl_ranking, 1):
        print(f"{i}. {model}: {score:.4f}")
    
    # Top Accuracy ranking (higher is better)
    print("\nRanking by Top Accuracy (higher is better):")
    acc_ranking = [("fine-tuned", results['avg_top_accuracy']['finetuned_vs_ground_truth'])]
    for model, score in results['avg_top_accuracy']['model_vs_ground_truth'].items():
        acc_ranking.append((model, score))
    acc_ranking.sort(key=lambda x: x[1], reverse=True)
    
    for i, (model, score) in enumerate(acc_ranking, 1):
        print(f"{i}. {model}: {score:.4f}")
    
    # Print category-wise metrics for fine-tuned model
    print("\n===== CATEGORY-WISE METRICS =====")
    for category, metrics in results["category_metrics"].items():
        print(f"\n{category.upper()}:")
        print(f"  Agreement: {metrics['agreement']:.4f}")
        print(f"  MSE: {metrics['mse']:.4f}")
        print(f"  Normalized Ranking Loss: {metrics['norm_ranking_loss']:.4f}")
        print(f"  Top Accuracy: {metrics['top_accuracy']:.4f}")
    
    # Calculate improvements over best baseline
    print("\n===== IMPROVEMENT ANALYSIS =====")
    
    # Find best baseline for each metric
    best_baseline_agreement = max(results['avg_agreement']['model_vs_ground_truth'].items(), key=lambda x: x[1])
    best_baseline_mse = min(results['avg_mse']['model_vs_ground_truth'].items(), key=lambda x: x[1])
    best_baseline_nrl = min(results['avg_norm_ranking_loss']['model_vs_ground_truth'].items(), key=lambda x: x[1])
    best_baseline_acc = max(results['avg_top_accuracy']['model_vs_ground_truth'].items(), key=lambda x: x[1])
    
    # Agreement (higher is better)
    finetuned_agreement = results['avg_agreement']['finetuned_vs_ground_truth']
    print(f"\nAgreement Comparison:")
    print(f"- Fine-tuned: {finetuned_agreement:.4f}")
    print(f"- Best baseline ({best_baseline_agreement[0]}): {best_baseline_agreement[1]:.4f}")
    
    if finetuned_agreement > best_baseline_agreement[1]:
        improvement = ((finetuned_agreement - best_baseline_agreement[1]) / best_baseline_agreement[1]) * 100
        print(f"- Improvement: +{improvement:.2f}%")
    else:
        difference = ((best_baseline_agreement[1] - finetuned_agreement) / best_baseline_agreement[1]) * 100
        print(f"- Difference: -{difference:.2f}%")
    
    # MSE (lower is better)
    finetuned_mse = results['avg_mse']['finetuned_vs_ground_truth']
    print(f"\nMSE Comparison:")
    print(f"- Fine-tuned: {finetuned_mse:.4f}")
    print(f"- Best baseline ({best_baseline_mse[0]}): {best_baseline_mse[1]:.4f}")
    
    if finetuned_mse < best_baseline_mse[1]:
        improvement = ((best_baseline_mse[1] - finetuned_mse) / best_baseline_mse[1]) * 100
        print(f"- Improvement: +{improvement:.2f}%")
    else:
        difference = ((finetuned_mse - best_baseline_mse[1]) / best_baseline_mse[1]) * 100
        print(f"- Difference: +{difference:.2f}%")
    
    # Normalized Ranking Loss (lower is better)
    finetuned_nrl = results['avg_norm_ranking_loss']['finetuned_vs_ground_truth']
    print(f"\nNormalized Ranking Loss Comparison:")
    print(f"- Fine-tuned: {finetuned_nrl:.4f}")
    print(f"- Best baseline ({best_baseline_nrl[0]}): {best_baseline_nrl[1]:.4f}")
    
    if finetuned_nrl < best_baseline_nrl[1]:
        improvement = ((best_baseline_nrl[1] - finetuned_nrl) / best_baseline_nrl[1]) * 100
        print(f"- Improvement: +{improvement:.2f}%")
    else:
        difference = ((finetuned_nrl - best_baseline_nrl[1]) / best_baseline_nrl[1]) * 100
        print(f"- Difference: +{difference:.2f}%")
    
    # Top Accuracy (higher is better)
    finetuned_acc = results['avg_top_accuracy']['finetuned_vs_ground_truth']
    print(f"\nTop Accuracy Comparison:")
    print(f"- Fine-tuned: {finetuned_acc:.4f}")
    print(f"- Best baseline ({best_baseline_acc[0]}): {best_baseline_acc[1]:.4f}")
    
    if finetuned_acc > best_baseline_acc[1]:
        improvement = ((finetuned_acc - best_baseline_acc[1]) / best_baseline_acc[1]) * 100
        print(f"- Improvement: +{improvement:.2f}%")
    else:
        difference = ((best_baseline_acc[1] - finetuned_acc) / best_baseline_acc[1]) * 100
        print(f"- Difference: -{difference:.2f}%")
    
    print(f"\nResults saved to {output_dir}") #Import your existing function