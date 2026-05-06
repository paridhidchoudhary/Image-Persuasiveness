import os
import base64
import json
import time
import numpy as np
from groq import Groq
from sklearn.model_selection import train_test_split
from collections import defaultdict
from PIL import Image
import random
import numpy as np
import torch  # If you're using PyTorch elsewhere
# Import the extract_text_data function
from simple_data_preprocess import extract_text_data

SEED = 50
random.seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

## SET PATHS
data_root = "/home/deepg/NAS/Downloads/MTP-2-persuasion/dataset"
dataset_image = os.path.join(data_root, "dataset_image")
dataset_response = os.path.join(data_root, "dataset_response")
dataset_user_preferred = os.path.join(data_root, "final_data")
output_dir = "./llama4_evaluation"
os.makedirs(output_dir, exist_ok=True)

# Configure Groq API
API_KEY = "gsk_xxxxxxxxx"  # Replace with your Groq API key
MODEL_NAME = "meta-llama/llama-4-scout-17b-16e-instruct"

# Initialize Groq client
client = Groq(api_key=API_KEY)

# Model mappings for comparison
MODEL_FILES = {
    "qwen_zeroshot": "output_qwen_zeroshot.txt",
    "qwen_fewshot": "output_qwen_fewshot.txt",
    "pixtral_zeroshot": "output_pixtral_zeroshot.txt",
    "pixtral_fewshot": "output_pixtral_fewshot.txt"
}

# Helper functions
def encode_image(image_path):
    """Encodes an image as a base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

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

def load_dataset():
    """Load all data from the dataset matching the same approach as other code"""
    data = []
    MAX_IMAGES = 4  # Maximum images per group

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
                    
                    # Check that ground truth file exists
                    user_preferred_path = os.path.join(dataset_user_preferred, category, group, "user_output.txt")
                    
                    if os.path.exists(user_preferred_path):
                        # Include this group
                        data.append({
                            "images": images,
                            "category": category,
                            "group": group,
                            "ground_truth_path": user_preferred_path
                        })

    print(f"Dataset loaded with {len(data)} groups having ground truth")
    return data

def get_llama_outputs(images, category, group):
    """Generate outputs from Llama model for a set of images"""
    # Define prompt
    prompt = f"You are evaluating images in the '{group}' group under the '{category}' product category. Rank the images, based on their appeal for selling '{category}' product. Provide description, and **persuasion score** between (1-100) for each image and explain the ranking."
    
    # Create content with text and images
    content = [{"type": "text", "text": prompt}]
    
    # Add each image to the content
    for img_path in images:
        try:
            base64_image = encode_image(img_path)
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            })
        except Exception as e:
            print(f"Error encoding image {img_path}: {e}")
            continue

    # Create messages list for the API call
    messages = [{"role": "user", "content": content}]

    # Add retry mechanism with exponential backoff
    max_retries = 3
    retry_delay = 2
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0,
                max_tokens=800,
                top_p=1.0,
                seed=50 
            )
            output_text = response.choices[0].message.content
            
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
            output_file = os.path.join(output_dir, f"{category}_{group}_llama_output.txt")
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(output_text)
                
            print(f"Generated Llama output saved to {output_file}")
            return scores, output_text
            
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Attempt {attempt+1} failed: {e}. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                print(f"Error generating Llama outputs for {category}/{group}: {e}")
                return None, None

def analyze_llama4_vs_ground_truth():
    """Compare Llama outputs to ground truth and other models"""
    # Load the dataset
    all_data = load_dataset()
    
    # Split the dataset using the same parameters as other code
    train_data, test_data = train_test_split(all_data, test_size=0.05, random_state=45)
    
    print(f"Validation set size: {len(test_data)} groups")
    
    # Initialize stats
    stats = {
        "total_samples": len(test_data),
        "llama_vs_ground_truth_agreements": [],
        "model_vs_ground_truth_agreements": defaultdict(list),
        "llama_vs_model_agreements": defaultdict(list),
        "llama_output_scores": [],
        "valid_samples": 0,
        "detailed_results": []
    }
    
    # Open file for detailed results
    details_file = os.path.join(output_dir, "detailed_results.txt")
    with open(details_file, "w", encoding="utf-8") as detailed_f:
        detailed_f.write("LLAMA MODEL EVALUATION ANALYSIS\n")
        detailed_f.write("=============================\n\n")
        
        # Process each sample in validation set
        for i, sample in enumerate(test_data):
            category = sample["category"]
            group = sample["group"]
            images = sample["images"]
            ground_truth_path = sample["ground_truth_path"]
            
            print(f"Processing {i+1}/{len(test_data)}: {category}/{group}")
            detailed_f.write(f"Sample {i+1}: {category}/{group}\n")
            
            # Extract ground truth scores
            ground_truth_scores, ground_truth_info = extract_scores(ground_truth_path)
            
            if not ground_truth_scores:
                print(f"Could not extract ground truth scores")
                detailed_f.write(f"Could not extract ground truth scores\n\n")
                continue
            
            # Compute ground truth ranking
            ground_truth_ranking = get_ranking_from_scores(ground_truth_scores, handle_ties=True)
            detailed_f.write(f"  Ground truth scores: {ground_truth_scores}\n")
            detailed_f.write(f"  Ground truth ranking: {ground_truth_ranking}\n")
            
            # Get Llama scores
            print(f"  Generating Llama output...")
            llama_scores, llama_output = get_llama_outputs(images, category, group)
            
            if not llama_scores:
                print(f"Could not generate scores from Llama model")
                detailed_f.write(f"Could not generate scores from Llama model\n\n")
                continue
                
            detailed_f.write(f"  Llama scores: {llama_scores}\n")
            stats["llama_output_scores"].append(llama_scores)
            
            # Ensure score lengths match
            if len(llama_scores) != len(ground_truth_scores):
                print(f"Score length mismatch: Llama={len(llama_scores)}, ground_truth={len(ground_truth_scores)}")
                detailed_f.write(f"Score length mismatch\n\n")
                continue
            
            # Get Llama ranking and calculate agreement with ground truth
            llama_ranking = get_ranking_from_scores(llama_scores, handle_ties=True)
            detailed_f.write(f"  Llama ranking: {llama_ranking}\n")
            
            # Calculate agreement between Llama and ground truth
            llama_vs_ground_truth = calculate_rank_agreement(ground_truth_ranking, llama_ranking)
            stats["llama_vs_ground_truth_agreements"].append(llama_vs_ground_truth)
            
            detailed_f.write(f"  Agreement between Llama and ground truth: {llama_vs_ground_truth:.4f}\n")
            
            # Compare with each baseline model
            valid_sample = True
            model_agreements = {}
            
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
                
                # Calculate agreement with ground truth
                ground_truth_agreement = calculate_rank_agreement(ground_truth_ranking, model_ranking)
                stats["model_vs_ground_truth_agreements"][model_name].append(ground_truth_agreement)
                
                # Calculate agreement with Llama
                llama_agreement = calculate_rank_agreement(llama_ranking, model_ranking)
                stats["llama_vs_model_agreements"][model_name].append(llama_agreement)
                
                model_agreements[model_name] = {
                    "scores": model_scores,
                    "ranking": model_ranking,
                    "ground_truth_agreement": ground_truth_agreement,
                    "llama_agreement": llama_agreement
                }
                
                detailed_f.write(f"  {model_name} scores: {model_scores}\n")
                detailed_f.write(f"  {model_name} ranking: {model_ranking}\n")
                detailed_f.write(f"  Agreement between {model_name} and ground truth: {ground_truth_agreement:.4f}\n")
                detailed_f.write(f"  Agreement between {model_name} and Llama: {llama_agreement:.4f}\n")
            
            # If all models had valid scores for this sample
            if valid_sample:
                stats["valid_samples"] += 1
                
                # Store detailed result
                stats["detailed_results"].append({
                    "category": category,
                    "group": group,
                    "ground_truth_scores": ground_truth_scores,
                    "ground_truth_ranking": ground_truth_ranking,
                    "llama_scores": llama_scores,
                    "llama_ranking": llama_ranking,
                    "llama_vs_ground_truth": llama_vs_ground_truth,
                    "model_agreements": model_agreements
                })
            
            detailed_f.write("\n" + "-" * 60 + "\n\n")
    
    # Calculate average agreements
    avg_llama_vs_ground_truth = np.mean(stats["llama_vs_ground_truth_agreements"]) if stats["llama_vs_ground_truth_agreements"] else 0
    
    avg_model_vs_ground_truth = {}
    for model, agreements in stats["model_vs_ground_truth_agreements"].items():
        if agreements:
            avg_model_vs_ground_truth[model] = np.mean(agreements)
        else:
            avg_model_vs_ground_truth[model] = 0
    
    avg_llama_vs_model = {}
    for model, agreements in stats["llama_vs_model_agreements"].items():
        if agreements:
            avg_llama_vs_model[model] = np.mean(agreements)
        else:
            avg_llama_vs_model[model] = 0
    
    # Return results
    return {
        "total_samples": stats["total_samples"],
        "valid_samples": stats["valid_samples"],
        "avg_llama_vs_ground_truth": avg_llama_vs_ground_truth,
        "avg_model_vs_ground_truth": avg_model_vs_ground_truth,
        "avg_llama_vs_model": avg_llama_vs_model,
        "detailed_results": stats["detailed_results"]
    }

def main():
    """Main function for Llama model evaluation"""
    print("Analyzing Llama model agreement with ground truth from user preferred outputs...")
    results = analyze_llama4_vs_ground_truth()
    
    # Save results to JSON
    with open(os.path.join(output_dir, "llama_agreement_results.json"), "w") as f:
        # Convert NumPy types to Python types for JSON serialization
        serializable_results = {
            "total_samples": int(results["total_samples"]),
            "valid_samples": int(results["valid_samples"]),
            "avg_llama_vs_ground_truth": float(results["avg_llama_vs_ground_truth"]),
            "avg_model_vs_ground_truth": {k: float(v) for k, v in results["avg_model_vs_ground_truth"].items()},
            "avg_llama_vs_model": {k: float(v) for k, v in results["avg_llama_vs_model"].items()},
        }
        json.dump(serializable_results, f, indent=2)
    
    # Print results
    print("\n===== LLAMA MODEL VS GROUND TRUTH ANALYSIS =====")
    print(f"Total validation samples: {results['total_samples']}")
    print(f"Valid samples with complete data: {results['valid_samples']}")
    
    print(f"\nAverage agreement between Llama and ground truth: {results['avg_llama_vs_ground_truth']:.4f}")
    
    print("\nAverage agreement between baseline models and ground truth:")
    for model, agreement in results["avg_model_vs_ground_truth"].items():
        print(f"- {model} vs ground truth: {agreement:.4f}")
    
    print("\nAverage agreement between Llama and baseline models:")
    for model, agreement in results["avg_llama_vs_model"].items():
        print(f"- Llama vs {model}: {agreement:.4f}")
    
    # Compare performance to baseline models
    print("\n===== PERFORMANCE COMPARISON =====")
    print("Rankings by agreement with ground truth:")
    
    all_models_avg = [("llama-model", results["avg_llama_vs_ground_truth"])]
    for model, agreement in results["avg_model_vs_ground_truth"].items():
        all_models_avg.append((model, agreement))
    
    # Sort by agreement (descending)
    all_models_avg.sort(key=lambda x: x[1], reverse=True)
    
    for i, (model, agreement) in enumerate(all_models_avg, 1):
        print(f"{i}. {model}: {agreement:.4f}")
    
    # Show specific comparison to best performing baseline model
    baseline_models = [(model, agr) for model, agr in results["avg_model_vs_ground_truth"].items()]
    if baseline_models:
        best_baseline = max(baseline_models, key=lambda x: x[1])
        llama_agreement = results["avg_llama_vs_ground_truth"]
        
        print(f"\nSpecific comparison:")
        print(f"- Llama model agreement with ground truth: {llama_agreement:.4f}")
        print(f"- Best baseline ({best_baseline[0]}) agreement with ground truth: {best_baseline[1]:.4f}")
        
        if llama_agreement > best_baseline[1]:
            improvement = ((llama_agreement - best_baseline[1]) / best_baseline[1]) * 100
            print(f"- Improvement: +{improvement:.2f}%")
        else:
            difference = ((best_baseline[1] - llama_agreement) / best_baseline[1]) * 100
            print(f"- Difference: -{difference:.2f}%")
    
    print(f"\nResults saved to {output_dir}")

if __name__ == "__main__":
    main()