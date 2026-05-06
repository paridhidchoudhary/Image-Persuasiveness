import os
import torch
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from collections import defaultdict
from transformers import AutoProcessor, AutoModelForImageTextToText
from huggingface_hub import login
from qwen_vl_utils import process_vision_info
import random
# Import your existing function
from simple_data_preprocess import extract_text_data
SEED = 50
random.seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    
## SET PATHS
data_root = "home/debajyoti/paridhi_mtp/product_images_real"
dataset_image = os.path.join(data_root, "dataset_image_new")

dataset_response = os.path.join(data_root, "dataset_response_new")
# Updated path to user preferred outputs (ground truth)
dataset_user_preferred = os.path.join(data_root, "final_data")
m1_outputs_dir = os.path.join(data_root, "m1_inference_outputs")
output_dir = "./final_model_evaluation"
os.makedirs(output_dir, exist_ok=True)

# Model mappings
MODEL_FILES = {
    "qwen_zeroshot": "output_qwen_zeroshot.txt",
    "qwen_fewshot": "output_qwen_fewshot.txt",
    "pixtral_zeroshot": "output_pixtral_zeroshot.txt",
    "pixtral_fewshot": "output_pixtral_fewshot.txt"
}


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
    FINETUNED_MODEL_NAME = "./vlm_finetuned_full_context_47_1_m2"
    print(f"Loading fine-tuned model: {FINETUNED_MODEL_NAME}")
    # Authenticate with Hugging Face
    login(token="hf_xxxxxxxxx")
    
    # Load model and processor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(FINETUNED_MODEL_NAME, use_fast=False)
    model = AutoModelForImageTextToText.from_pretrained(FINETUNED_MODEL_NAME).to(device)
    
    print(f"Model and processor loaded successfully on {device}!")
    return model, processor, device

def get_model_outputs(model, processor, device, images, category, group,m1_output):
    """Generate and extract scores from the fine-tuned model for a set of images"""
    try:
        # Format image paths for processing
        sorted_images = []
        for img_path in images:
            sorted_images.append({"type": "image", "image": f"file://{img_path}"})
        
        messages = [
            {
                "role": "user",
                "content": sorted_images + [
                    {
                        "type": "text", 
                        "text": f"You are evaluating images in the '{group}' group under the '{category}' product category. "
                                f"Rank the images, based on their appeal for selling '{category}' product. "
                                f"Provide description, and **persuasion score (1-100)** for each image and explain the ranking."
                    }
                ],
            },
            {
                "role": "assistant",
                "content": m1_output  # Include M1's output as context
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Review and refine the above ranking and explanation to make it more accurate and insightful."
                    }
                ]
            }
        ]
        
        # Process the inputs
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(device)

        # Generate response
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=800)

        # Decode output
        output_text = processor.batch_decode(output, skip_special_tokens=True)[0]
        output_text = output_text.split("assistant")[-1].strip()
        # Write response to a temporary file for extract_text_data
        temp_file = os.path.join(output_dir, f"temp_{category}_{group}.txt")
        with open(temp_file, "w", encoding="utf-8") as f:
            f.write(output_text)
        
        # Extract information using extract_text_data function
        extracted_info, _ = extract_text_data(temp_file)
        
        if not extracted_info:
            print(f"Failed to extract information from model response for {category}/{group}")
            return None, output_text
        
        # Extract scores
        extracted_info.sort(key=lambda x: x["image_num"])
        scores = [item.get("score", 0) for item in extracted_info]
        
        if not all(score is not None for score in scores):
            print(f"Missing scores in model response for {category}/{group}")
            return None, output_text
        
        # Save the full output
        output_file = os.path.join(output_dir, f"{category}_{group}_finetuned_output.txt")
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(output_text)
            
        print(f"Generated finetuned model output saved to {output_file}")
        return scores, output_text
        
    except Exception as e:
        print(f"Error in get_model_outputs for {category}/{group}: {e}")
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
                    m1_output_path = os.path.join(m1_outputs_dir, category, group, "m1_output.txt")
                    user_preferred_path = os.path.join(dataset_user_preferred, category, group, "user_output.txt")
                    try:
                        with open(m1_output_path, 'r') as f:
                                m1_output = f.read()
                    except:
                        continue
                    if os.path.exists(user_preferred_path):
                        # Include this group without extracting scores yet
                        data.append({
                            "images": images,
                            "category": category,
                            "group": group,
                            "m1_output": m1_output,
                            "ground_truth_path": user_preferred_path
                        })

    print(f"Dataset loaded with {len(data)} groups having ground truth")
    return data

def analyze_model_performance():
    """Comprehensive evaluation of fine-tuned model against ground truth and baseline models"""
    # Load the dataset with ground truth
    all_data = load_dataset()
    
    # Split the dataset (use the same random_state as in your training)
    train_data, test_data = train_test_split(all_data, test_size=0.05, random_state=47)
    
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
            m1_output = sample["m1_output"]
            print(f"Processing {i+1}/{len(test_data)}: {category}/{group}")
            detailed_f.write(f"Sample {i+1}: {category}/{group}\n")
            
            # Extract ground truth scores from file path
            ground_truth_scores, ground_truth_info = extract_scores(ground_truth_path)
            
            if not ground_truth_scores:
                print(f"Could not extract ground truth scores")
                detailed_f.write(f"Could not extract ground truth scores\n\n")
                continue
            
            # Compute ground truth ranking
            ground_truth_ranking = get_ranking_from_scores(ground_truth_scores, handle_ties=True)
            detailed_f.write(f"  Ground truth scores: {ground_truth_scores}\n")
            detailed_f.write(f"  Ground truth ranking: {ground_truth_ranking}\n")
            
            # Get fine-tuned model scores by running inference
            print(f"  Generating finetuned model output...")
            finetuned_scores, finetuned_output = get_model_outputs(model, processor, device, images, category, group , m1_output)
            
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
            
            # Compare with each baseline model
            valid_sample = True
            model_metrics = {}
            
            for model_name, file_name in MODEL_FILES.items():
                model_path = os.path.join(dataset_response, category, group, file_name)
                if not os.path.exists(model_path):
                    print(f"  Missing file for model {model_name}")
                    valid_sample = False
                    continue
                
                model_scores, _ = extract_scores(model_path)
                if not model_scores:
                    print(f"  Could not extract scores for model {model_name}")
                    valid_sample = False
                    continue
                
                # Check if score lengths match
                if len(model_scores) != len(ground_truth_scores):
                    print(f"  Score length mismatch for model {model_name}: {len(model_scores)} vs {len(ground_truth_scores)}")
                    valid_sample = False
                    continue
                
                # Get model ranking
                model_ranking = get_ranking_from_scores(model_scores, handle_ties=True)
                
                # Calculate metrics with ground truth
                gt_agreement = calculate_rank_agreement(ground_truth_ranking, model_ranking)
                gt_mse = calculate_mse(ground_truth_scores, model_scores)
                gt_norm_rank_loss = calculate_normalized_ranking_loss(ground_truth_ranking, model_ranking)
                gt_top_acc = calculate_top_accuracy(ground_truth_ranking, model_ranking)
                
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
                
                model_metrics[model_name] = {
                    "scores": model_scores,
                    "ranking": model_ranking,
                    "vs_ground_truth": {
                        "agreement": gt_agreement,
                        "mse": gt_mse,
                        "norm_ranking_loss": gt_norm_rank_loss,
                        "top_accuracy": gt_top_acc,
                    },
                    "vs_finetuned": {
                        "agreement": ft_agreement,
                        "mse": ft_mse,
                        "norm_ranking_loss": ft_norm_rank_loss,
                        "top_accuracy": ft_top_acc,
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
            
            # If all models had valid scores for this sample
            if valid_sample:
                stats["valid_samples"] += 1
                
                # Store detailed result
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
                        "model_metrics": model_metrics
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
    
    # --- ADD THIS SECTION TO PRINT THE VECTORS ---

    print("\n===== PER-SAMPLE METRIC VECTORS VS GROUND TRUTH =====")
    print(f"(These vectors correspond to the {raw_stats['valid_samples']} valid samples)")

    # Fine-tuned model vectors
    print("\nFine-tuned Model vs Ground Truth:")
    # Format agreement for readability
    print("  Agreement Vector:", [f"{x:.4f}" for x in raw_stats["agreement"]["finetuned_vs_ground_truth"]])
    # Top accuracy is 0 or 1, print directly
    print("  Top Accuracy Vector:", raw_stats["top_accuracy"]["finetuned_vs_ground_truth"])

    # Baseline model vectors
    # Iterate through the keys that actually exist in the results (only for valid samples)
    for model_name in raw_stats["agreement"]["model_vs_ground_truth"].keys():
         print(f"\n{model_name} vs Ground Truth:")
         print("  Agreement Vector:", [f"{x:.4f}" for x in raw_stats["agreement"]["model_vs_ground_truth"][model_name]])
         print("  Top Accuracy Vector:", raw_stats["top_accuracy"]["model_vs_ground_truth"][model_name])


    # --- END ADD THIS SECTION ---


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