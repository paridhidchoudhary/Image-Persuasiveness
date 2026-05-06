import os
import re
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from collections import defaultdict
from transformers import AutoProcessor, AutoModelForImageTextToText
from huggingface_hub import login
from qwen_vl_utils import process_vision_info
import random 
import torch 
import numpy as np
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
# Import your existing function
from simple_data_preprocess import extract_text_data

## SET PATHS
# Define paths (adjust as needed)
data_root = "home/debajyoti/paridhi_mtp/product_images_real"
#data_root = "/home/deepg/NAS/Downloads/MTP-2-persuasion/dataset"
dataset_image = os.path.join(data_root, "dataset_image_new")
dataset_response = os.path.join(data_root, "dataset_response_new")
dataset_response_user = os.path.join(data_root, "dataset_response copy")
user_output_file = "real_user_output.txt"
output_dir = "./finetuned_model_evaluation"
os.makedirs(output_dir, exist_ok=True)

# Model mappings (from your code)
# MODEL_NAMES = {
#     1: "qwen_zeroshot",
#     2: "pixtral_zeroshot",
#     3: "qwen_fewshot",
#     4: "pixtral_fewshot"
# }

MODEL_NAMES = {
    1: "qwen_zeroshot",
    2: "qwen_fewshot",
    3: "pixtral_zeroshot",
    4: "pixtral_fewshot"
}

MODEL_FILES = {
    "qwen_zeroshot": "output_qwen_zeroshot.txt",
    "qwen_fewshot": "output_qwen_fewshot.txt",
    "pixtral_zeroshot": "output_pixtral_zeroshot.txt",
    "pixtral_fewshot": "output_pixtral_fewshot.txt"
}

# Fine-tuned model information
FINETUNED_MODEL_NAME = "Deb123/qwen2.5-vl-7b-pair-finetuned-private"  # Update to your model name
FINETUNED_MODEL_NAME = "Deb123/qwen2.5-vl-7b-pair-finetuned-smalltrain-private"

def parse_user_preferences(content):
    """Parse model preferences from user output file content"""
    preferences = {}
    model_match = re.search(r'model\s*:\s*([^\n]+)', content)
    if model_match:
        model_pref_str = model_match.group(1).strip()
        if "=" in model_pref_str:
            equal_models = [int(m.strip()) for m in model_pref_str.split("=")]
            preferences["top_model"] = equal_models[0]
            preferences["equal_models"] = equal_models
        else:
            try:
                ranked_models = [int(m.strip()) for m in model_pref_str.split(",")]
                preferences["top_model"] = ranked_models[0]
                preferences["ranked_models"] = ranked_models
            except ValueError:
                if model_pref_str.strip():
                    print(f"Invalid model preference: {model_pref_str}")
                    preferences["top_model"] = 1
    return preferences

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
    
    # Authenticate with Hugging Face
    login(token='hf_xxxxxxxxxx')  # Update with your token if needed
    
    # Load model and processor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(FINETUNED_MODEL_NAME, use_fast=False)
    model = AutoModelForImageTextToText.from_pretrained(FINETUNED_MODEL_NAME).to(device)
    
    print(f"Model and processor loaded successfully on {device}!")
    return model, processor, device

def resize_if_needed(img, max_size=384):
    """Resize an image if its dimensions exceed max_size while maintaining aspect ratio."""
    if img.width > max_size or img.height > max_size:
        scaling_factor = max_size / float(max(img.width, img.height))
        new_width = int(img.width * scaling_factor)
        new_height = int(img.height * scaling_factor)
        return img.resize((new_width, new_height), Image.LANCZOS)
    return img

def get_model_outputs(model, processor, device, images, category, group):
    """Generate and extract scores from the fine-tuned model for a set of images"""
    try:
        # Format image paths for processing
        formatted_images = []
        for img_path in images:
            formatted_images.append({"type": "image", "image": f"file://{img_path}"})
        
        # Create user message
        messages = [
            {
                "role": "user",
                "content": formatted_images + [
                    {
                        "type": "text",
                        "text": f"You are evaluating images in the '{group}' group under the '{category}' product category. "
                                f"Rank the images, based on their appeal for selling '{category}' product. "
                                f"Provide description, and **persuasion score (1-100)** for each image and explain the ranking."
                    }
                ],
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
        
        return scores, output_text
        
    except Exception as e:
        print(f"Error in get_model_outputs for {category}/{group}: {e}")
        return None, None

def load_dataset():
    """Load all data from the dataset"""
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
                    
                    # Check for user preference
                    user_pref_path = os.path.join(dataset_response_user, category, group, user_output_file)
                    user_preference = None
                    
                    if os.path.exists(user_pref_path):
                        try:
                            with open(user_pref_path, "r", encoding="utf-8") as f:
                                user_content = f.read()
                            
                            preferences = parse_user_preferences(user_content)
                            if "top_model" in preferences:
                                user_preference = preferences["top_model"]
                        except Exception as e:
                            print(f"Error parsing user preference for {category}/{group}: {e}")
                    
                    # Only keep entries with user preferences
                    if user_preference is not None:
                        data.append({
                            "images": images,
                            "category": category,
                            "group": group,
                            "user_preference": user_preference
                        })

    print(f"Dataset loaded with {len(data)} groups having user preferences")
    return data

def analyze_finetuned_vs_user():
    """Compare the fine-tuned model outputs to user preferences"""
    # Load the dataset with user preferences
    all_data = load_dataset()
    
    # Split the dataset (use the same random_state as in your training)
    train_data, val_data = train_test_split(all_data, test_size=0.05, random_state=42)
    
    print(f"Validation set size: {len(val_data)} groups")
    
    # Initialize the fine-tuned model
    model, processor, device = initialize_model()
    
    # Initialize stats
    stats = {
        "total_samples": len(val_data),
        "model_preferences": {
            "qwen_zeroshot": 0,
            "pixtral_zeroshot": 0,
            "qwen_fewshot": 0,
            "pixtral_fewshot": 0
        },
        "finetuned_vs_user_agreements": [],
        "finetuned_vs_user_agreements_by_pref_model": defaultdict(list),
        "finetuned_vs_model_agreements": defaultdict(list),
        "baseline_vs_user_agreements": defaultdict(list),
        "finetuned_output_scores": [],
        "valid_samples": 0,
        "detailed_results": []
    }
    
    # Open file for detailed results
    details_file = os.path.join(output_dir, "detailed_results.txt")
    with open(details_file, "w", encoding="utf-8") as detailed_f:
        detailed_f.write("DETAILED AGREEMENT ANALYSIS\n")
        detailed_f.write("==========================\n\n")
        
        # Process each sample in validation set
        for i, sample in enumerate(val_data):
            category = sample["category"]
            group = sample["group"]
            images = sample["images"]
            user_pref_num = sample["user_preference"]
            user_pref_model = MODEL_NAMES.get(user_pref_num)
            
            print(f"Processing {i+1}/{len(val_data)}: {category}/{group}")
            detailed_f.write(f"Sample {i+1}: {category}/{group}\n")
            detailed_f.write(f"User preferred model: {user_pref_model} (#{user_pref_num})\n")
            
            if not user_pref_model:
                print(f"Unknown model preference: {user_pref_num} for {category}/{group}")
                detailed_f.write("Unknown model preference\n\n")
                continue
            
            # Increment the model preference count
            stats["model_preferences"][user_pref_model] += 1
            
            # Get preferred model scores
            preferred_model_path = os.path.join(dataset_response, category, group, MODEL_FILES[user_pref_model])
            if not os.path.exists(preferred_model_path):
                print(f"Missing file for preferred model {user_pref_model} in {category}/{group}")
                detailed_f.write(f"Missing file for preferred model\n\n")
                continue
            
            preferred_scores, _ = extract_scores(preferred_model_path)
            if not preferred_scores:
                print(f"Could not extract scores for preferred model {user_pref_model} in {category}/{group}")
                detailed_f.write(f"Could not extract scores for preferred model\n\n")
                continue
            
            detailed_f.write(f"  Preferred model scores: {preferred_scores}\n")
            
            # Get fine-tuned model scores
            finetuned_scores, finetuned_output = get_model_outputs(model, processor, device, images, category, group)
            if not finetuned_scores:
                print(f"Could not generate scores from fine-tuned model for {category}/{group}")
                detailed_f.write(f"Could not generate scores from fine-tuned model\n\n")
                continue
                
            detailed_f.write(f"  Fine-tuned model scores: {finetuned_scores}\n")
            stats["finetuned_output_scores"].append(finetuned_scores)
            
            # Ensure score lengths match
            if len(finetuned_scores) != len(preferred_scores):
                print(f"Score length mismatch: finetuned={len(finetuned_scores)}, preferred={len(preferred_scores)}")
                detailed_f.write(f"  ⚠️ Score length mismatch\n\n")
                continue
            
            # Get rankings and calculate agreement with user preferred model
            preferred_ranking = get_ranking_from_scores(preferred_scores, handle_ties=True)
            finetuned_ranking = get_ranking_from_scores(finetuned_scores, handle_ties=True)
            
            detailed_f.write(f"  Preferred model ranking: {preferred_ranking}\n")
            detailed_f.write(f"  Fine-tuned model ranking: {finetuned_ranking}\n")
            
            # Calculate agreement between fine-tuned model and user preferred model
            finetuned_vs_user_agreement = calculate_rank_agreement(preferred_ranking, finetuned_ranking)
            stats["finetuned_vs_user_agreements"].append(finetuned_vs_user_agreement)
            stats["finetuned_vs_user_agreements_by_pref_model"][user_pref_model].append(finetuned_vs_user_agreement)
            
            detailed_f.write(f"  Agreement between fine-tuned model and user preferred model: {finetuned_vs_user_agreement:.4f}\n")
            
            # Compare with each baseline model
            valid_sample = True
            model_agreements = {}
            
            for model_name, file_name in MODEL_FILES.items():
                model_path = os.path.join(dataset_response, category, group, file_name)
                if not os.path.exists(model_path):
                    print(f"Missing file for model {model_name} in {category}/{group}")
                    valid_sample = False
                    continue
                
                model_scores, _ = extract_scores(model_path)
                if not model_scores:
                    print(f"Could not extract scores for model {model_name} in {category}/{group}")
                    valid_sample = False
                    continue
                
                # Check if score lengths match
                if len(model_scores) != len(preferred_scores):
                    print(f"Score length mismatch for model {model_name} in {category}/{group}")
                    valid_sample = False
                    continue
                
                # Get model ranking
                model_ranking = get_ranking_from_scores(model_scores, handle_ties=True)
                
                # Calculate agreement with user preferred model
                user_agreement = calculate_rank_agreement(preferred_ranking, model_ranking)
                stats["baseline_vs_user_agreements"][model_name].append(user_agreement)
                
                # Calculate agreement with fine-tuned model
                finetuned_agreement = calculate_rank_agreement(finetuned_ranking, model_ranking)
                stats["finetuned_vs_model_agreements"][model_name].append(finetuned_agreement)
                
                model_agreements[model_name] = {
                    "scores": model_scores,
                    "ranking": model_ranking,
                    "user_agreement": user_agreement,
                    "finetuned_agreement": finetuned_agreement
                }
                
                detailed_f.write(f"  {model_name} scores: {model_scores}\n")
                detailed_f.write(f"  {model_name} ranking: {model_ranking}\n")
                detailed_f.write(f"  Agreement between {model_name} and user preferred model: {user_agreement:.4f}\n")
                detailed_f.write(f"  Agreement between {model_name} and fine-tuned model: {finetuned_agreement:.4f}\n")
            
            # If all models had valid scores for this sample
            if valid_sample:
                stats["valid_samples"] += 1
                
                # Store detailed result
                stats["detailed_results"].append({
                    "category": category,
                    "group": group,
                    "user_preference": user_pref_model,
                    "preferred_scores": preferred_scores,
                    "preferred_ranking": preferred_ranking,
                    "finetuned_scores": finetuned_scores,
                    "finetuned_ranking": finetuned_ranking,
                    "finetuned_vs_user_agreement": finetuned_vs_user_agreement,
                    "model_agreements": model_agreements
                })
            
            detailed_f.write("\n" + "-" * 60 + "\n\n")
            
            # Save fine-tuned model output
            finetuned_output_file = os.path.join(output_dir, f"{category}_{group}_finetuned_output.txt")
            if finetuned_output:
                with open(finetuned_output_file, "w", encoding="utf-8") as f:
                    f.write(finetuned_output)
    
    # Calculate average agreements
    avg_finetuned_vs_user = np.mean(stats["finetuned_vs_user_agreements"]) if stats["finetuned_vs_user_agreements"] else 0
    
    avg_finetuned_vs_user_by_pref = {}
    for model, agreements in stats["finetuned_vs_user_agreements_by_pref_model"].items():
        if agreements:
            avg_finetuned_vs_user_by_pref[model] = np.mean(agreements)
        else:
            avg_finetuned_vs_user_by_pref[model] = 0
    
    avg_finetuned_vs_model = {}
    for model, agreements in stats["finetuned_vs_model_agreements"].items():
        if agreements:
            avg_finetuned_vs_model[model] = np.mean(agreements)
        else:
            avg_finetuned_vs_model[model] = 0
    
    avg_baseline_vs_user = {}
    for model, agreements in stats["baseline_vs_user_agreements"].items():
        if agreements:
            avg_baseline_vs_user[model] = np.mean(agreements)
        else:
            avg_baseline_vs_user[model] = 0
    
    # Return results
    return {
        "total_samples": stats["total_samples"],
        "valid_samples": stats["valid_samples"],
        "model_preferences": stats["model_preferences"],
        "avg_finetuned_vs_user": avg_finetuned_vs_user,
        "avg_finetuned_vs_user_by_pref": avg_finetuned_vs_user_by_pref,
        "avg_finetuned_vs_model": avg_finetuned_vs_model,
        "avg_baseline_vs_user": avg_baseline_vs_user,
        "detailed_results": stats["detailed_results"]
    }

if __name__ == "__main__":
    print("Analyzing fine-tuned model agreement with user preferences...")
    results = analyze_finetuned_vs_user()
    
    # Save results to JSON
    import json
    with open(os.path.join(output_dir, "finetuned_agreement_results.json"), "w") as f:
        # Convert NumPy types to Python types for JSON serialization
        serializable_results = {
            "total_samples": int(results["total_samples"]),
            "valid_samples": int(results["valid_samples"]),
            "model_preferences": {k: int(v) for k, v in results["model_preferences"].items()},
            "avg_finetuned_vs_user": float(results["avg_finetuned_vs_user"]),
            "avg_finetuned_vs_user_by_pref": {k: float(v) for k, v in results["avg_finetuned_vs_user_by_pref"].items()},
            "avg_finetuned_vs_model": {k: float(v) for k, v in results["avg_finetuned_vs_model"].items()},
            "avg_baseline_vs_user": {k: float(v) for k, v in results["avg_baseline_vs_user"].items()}
        }
        json.dump(serializable_results, f, indent=2)
    
    # Print results
    print("\n===== FINE-TUNED MODEL VS USER PREFERENCE ANALYSIS =====")
    print(f"Total validation samples: {results['total_samples']}")
    print(f"Valid samples with complete data: {results['valid_samples']}")
    
    print("\nModel Preference Distribution:")
    for model, count in results["model_preferences"].items():
        percentage = (count / results["total_samples"]) * 100 if results["total_samples"] > 0 else 0
        print(f"- {model}: {count} samples ({percentage:.2f}%)")
    
    print(f"\nAverage agreement between fine-tuned model and user preferred model: {results['avg_finetuned_vs_user']:.4f}")
    
    print("\nAverage agreement between fine-tuned model and each preferred model:")
    for model, agreement in results["avg_finetuned_vs_user_by_pref"].items():
        print(f"- When {model} is preferred: {agreement:.4f}")
    
    print("\nAverage agreement between fine-tuned model and each baseline model:")
    for model, agreement in results["avg_finetuned_vs_model"].items():
        print(f"- Fine-tuned vs {model}: {agreement:.4f}")
    
    print("\nAverage agreement between baseline models and user preferences:")
    for model, agreement in results["avg_baseline_vs_user"].items():
        print(f"- {model} vs user preference: {agreement:.4f}")
    
    # Compare performance to baseline models
    print("\n===== PERFORMANCE COMPARISON =====")
    print("Rankings by agreement with user preferences:")
    
    all_models_avg = [("fine-tuned model", results["avg_finetuned_vs_user"])]
    for model, agreement in results["avg_baseline_vs_user"].items():
        all_models_avg.append((model, agreement))
    
    # Sort by agreement (descending)
    all_models_avg.sort(key=lambda x: x[1], reverse=True)
    
    for i, (model, agreement) in enumerate(all_models_avg, 1):
        print(f"{i}. {model}: {agreement:.4f}")
    
    # Show specific comparison to pixtral_zeroshot
    pixtral_zero_agreement = results["avg_baseline_vs_user"].get("pixtral_zeroshot", 0)
    finetuned_agreement = results["avg_finetuned_vs_user"]
    
    print(f"\nSpecific comparison:")
    print(f"- Fine-tuned model agreement with user preferences: {finetuned_agreement:.4f}")
    print(f"- pixtral_zeroshot agreement with user preferences: {pixtral_zero_agreement:.4f}")
    
    if finetuned_agreement > pixtral_zero_agreement:
        improvement = ((finetuned_agreement - pixtral_zero_agreement) / pixtral_zero_agreement) * 100
        print(f"- Improvement: +{improvement:.2f}%")
    else:
        difference = ((pixtral_zero_agreement - finetuned_agreement) / pixtral_zero_agreement) * 100
        print(f"- Difference: -{difference:.2f}%")
        
    print(f"\nResults saved to {output_dir}")