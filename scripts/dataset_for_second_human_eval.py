import os
import json
import shutil
import re
import numpy as np
from collections import defaultdict

# Try to import the extract_text_data function
try:
    from simple_preprocess_4 import extract_text_data
except ImportError:
    # If not available, define a simplified version
    print("Warning: Could not import extract_text_data. Using simplified version.")
    
    def extract_text_data(file_path):
        """Simplified version to extract scores from model outputs"""
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Extract scores using regex patterns
        scores = []
        score_patterns = [
            r'Persuasion Score:\s*(\d+)/100',
            r'Persuasion Score:\s*(\d+)',
            r'Score:\s*(\d+)/100',
            r'Score:\s*(\d+)'
        ]
        
        # Try each pattern
        for pattern in score_patterns:
            matches = re.findall(pattern, content)
            if matches:
                scores = [int(score) for score in matches]
                break
        
        # Create simple extracted_info structure
        extracted_info = []
        for i, score in enumerate(scores, 1):
            extracted_info.append({
                "image_num": i,
                "score": score
            })
        
        return extracted_info, content


## SET PATHS
data_root = "home/debajyoti/paridhi_mtp/product_images_real"
#data_root = "/home/deepg/NAS/Downloads/MTP-2-persuasion/dataset"
dataset_image = os.path.join(data_root, "dataset_image_new")
#dataset_image = os.path.join(data_root, "dataset_image copy")
dataset_response = os.path.join(data_root, "dataset_response_new")
#dataset_response = os.path.join(data_root, "dataset_response copy")
dataset_response_user = os.path.join(data_root, "dataset_response copy")
user_output_file = "real_user_output.txt"

# Define output directory
output_dir = os.path.join(data_root, "human_verification_subset")
output_image_dir = os.path.join(output_dir, "images")
output_response_dir = os.path.join(output_dir, "responses")
output_stats_file = os.path.join(output_dir, "disagreement_stats.json")

# Create output directories
os.makedirs(output_dir, exist_ok=True)
os.makedirs(output_image_dir, exist_ok=True)
os.makedirs(output_response_dir, exist_ok=True)

# Define model names and file mappings
MODEL_NAMES = {
    1: "qwen_zeroshot",
    2: "pixtral_zeroshot",
    3: "qwen_fewshot",
    4: "pixtral_fewshot"
}

MODEL_FILES = {
    "qwen_zeroshot": "output_qwen_zeroshot.txt",
    "qwen_fewshot": "output_qwen_fewshot.txt", 
    "pixtral_zeroshot": "output_pixtral_zeroshot.txt",
    "pixtral_fewshot": "output_pixtral_fewshot.txt"
}

def get_ranking_from_scores(scores):
    """
    Convert scores to rankings (higher score = better rank)
    Handle ties by giving the same rank to tied scores
    """
    if not scores:
        return []
    
    # Create (index, score) pairs and sort by score (descending)
    pairs = [(i, score) for i, score in enumerate(scores)]
    sorted_pairs = sorted(pairs, key=lambda x: x[1], reverse=True)
    
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

def get_low_agreement_groups():
    """
    Find groups where model outputs have low agreement with user preferred model.
    Only includes groups where agreement is < 50%.
    """
    print("Analyzing groups to identify those with low agreement with user response...")
    
    low_agreement_groups = []
    stats_by_group = {}
    processed_count = 0
    error_count = 0
    
    # Process all categories and groups
    for category in os.listdir(dataset_image):
        category_path = os.path.join(dataset_image, category)
        if not os.path.isdir(category_path):
            continue
            
        for group in os.listdir(category_path):
            group_path = os.path.join(dataset_image, category, group)
            if not os.path.isdir(group_path):
                continue
                
            processed_count += 1
            if processed_count % 50 == 0:
                print(f"Processed {processed_count} groups so far...")
            
            # First check if user response exists
            user_response_path = os.path.join(dataset_response_user, category, group, user_output_file)
            if not os.path.exists(user_response_path):
                continue
                
            # Read user response
            try:
                with open(user_response_path, 'r', encoding='utf-8') as f:
                    user_content = f.read()
                
                # Parse user preference
                preferences = parse_user_preferences(user_content)
                if "top_model" not in preferences:
                    continue
                    
                top_model_num = preferences["top_model"]
                top_model_name = MODEL_NAMES.get(top_model_num)
                
                if not top_model_name:
                    continue
            except Exception as e:
                print(f"Error parsing user response for {category}/{group}: {str(e)}")
                error_count += 1
                continue
            
            # Get preferred model output
            preferred_model_path = os.path.join(dataset_response, category, group, MODEL_FILES[top_model_name])
            if not os.path.exists(preferred_model_path):
                continue
                
            try:
                # Extract scores from preferred model
                preferred_info, _ = extract_text_data(preferred_model_path)
                
                if not preferred_info or not all(item.get("score") is not None for item in preferred_info):
                    continue
                    
                # Get scores from preferred model
                preferred_info_sorted = sorted(preferred_info, key=lambda x: x["image_num"])
                preferred_scores = [item["score"] for item in preferred_info_sorted]
                
                # Get ranking from preferred model scores
                preferred_ranking = get_ranking_from_scores(preferred_scores)
                
                # Collect agreement with other models
                model_agreements = {}
                other_model_scores = {}
                all_agreements = []
                
                for model_name, file_name in MODEL_FILES.items():
                    if model_name == top_model_name:
                        # Skip the preferred model (agreement would be 100%)
                        continue
                        
                    model_path = os.path.join(dataset_response, category, group, file_name)
                    if not os.path.exists(model_path):
                        continue
                        
                    # Extract scores from this model
                    model_info, _ = extract_text_data(model_path)
                    
                    if not model_info or not all(item.get("score") is not None for item in model_info):
                        continue
                        
                    # Get scores from this model
                    model_info_sorted = sorted(model_info, key=lambda x: x["image_num"])
                    model_scores = [item["score"] for item in model_info_sorted]
                    
                    if len(model_scores) != len(preferred_scores):
                        continue
                        
                    # Store scores
                    other_model_scores[model_name] = model_scores
                    
                    # Get ranking from this model's scores
                    model_ranking = get_ranking_from_scores(model_scores)
                    
                    # Calculate agreement
                    agreement = calculate_rank_agreement(preferred_ranking, model_ranking)
                    model_agreements[model_name] = agreement
                    all_agreements.append(agreement)
                
                # Only continue if we have agreements with at least one other model
                if not all_agreements:
                    continue
                    
                # Calculate average agreement
                avg_agreement = np.mean(all_agreements)
                
                # If average agreement is below 50%, include this group
                if avg_agreement < 0.5:
                    group_info = {
                        "category": category,
                        "group": group,
                        "preferred_model": top_model_name,
                        "preferred_scores": preferred_scores,
                        "average_agreement": float(avg_agreement),
                        "model_agreements": model_agreements,
                        "other_model_scores": other_model_scores
                    }
                    
                    low_agreement_groups.append(group_info)
                    
                    # Store stats for this group
                    stats_by_group[f"{category}/{group}"] = {
                        "preferred_model": top_model_name,
                        "preferred_scores": preferred_scores,
                        "average_agreement": float(avg_agreement),
                        "model_agreements": {model: float(agreement) for model, agreement in model_agreements.items()}
                    }
            except Exception as e:
                print(f"Error processing {category}/{group}: {str(e)}")
                error_count += 1
                continue
    
    print(f"Analysis complete. Processed {processed_count} groups with {error_count} errors.")
    print(f"Found {len(low_agreement_groups)} groups with average agreement < 50% compared to user preferred model")
    
    # Save stats for low agreement groups
    with open(output_stats_file, "w", encoding="utf-8") as f:
        json.dump({
            "total_low_agreement_groups": len(low_agreement_groups),
            "groups": stats_by_group
        }, f, indent=2)
    
    return low_agreement_groups

def copy_group_data(low_agreement_groups):
    """
    Copy both images and model responses for the identified low agreement groups
    to the output directory.
    """
    print("Copying data for low agreement groups...")
    
    # Track statistics
    stats = {
        "successful_copies": 0,
        "image_errors": 0,
        "response_errors": 0,
        "user_response_errors": 0
    }
    
    for group_info in low_agreement_groups:
        category = group_info["category"]
        group = group_info["group"]
        
        print(f"Processing {category}/{group} (Agreement with preferred model: {group_info['average_agreement']:.2f})")
        
        # Create output directories for this group
        group_image_dir = os.path.join(output_image_dir, category, group)
        group_response_dir = os.path.join(output_response_dir, category, group)
        
        os.makedirs(group_image_dir, exist_ok=True)
        os.makedirs(group_response_dir, exist_ok=True)
        
        # Copy images
        source_image_dir = os.path.join(dataset_image, category, group)
        if os.path.exists(source_image_dir):
            try:
                for image_file in os.listdir(source_image_dir):
                    if image_file.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                        source_path = os.path.join(source_image_dir, image_file)
                        dest_path = os.path.join(group_image_dir, image_file)
                        shutil.copy2(source_path, dest_path)
                
                # Save agreement info
                with open(os.path.join(group_image_dir, "agreement_info.json"), "w", encoding="utf-8") as f:
                    # Create a clean version for saving
                    clean_info = {
                        "category": category,
                        "group": group,
                        "preferred_model": group_info["preferred_model"],
                        "average_agreement": float(group_info["average_agreement"]),
                        "model_agreements": {model: float(agreement) for model, agreement in group_info["model_agreements"].items()}
                    }
                    json.dump(clean_info, f, indent=2)
            except Exception as e:
                print(f"Error copying images for {category}/{group}: {e}")
                stats["image_errors"] += 1
                continue
        else:
            print(f"Image directory not found: {source_image_dir}")
            stats["image_errors"] += 1
            continue
        
        # Copy model responses
        source_response_dir = os.path.join(dataset_response, category, group)
        if os.path.exists(source_response_dir):
            try:
                for model_name, file_name in MODEL_FILES.items():
                    source_path = os.path.join(source_response_dir, file_name)
                    if os.path.exists(source_path):
                        dest_path = os.path.join(group_response_dir, file_name)
                        shutil.copy2(source_path, dest_path)
            except Exception as e:
                print(f"Error copying responses for {category}/{group}: {e}")
                stats["response_errors"] += 1
                continue
        else:
            print(f"Response directory not found: {source_response_dir}")
            stats["response_errors"] += 1
            continue
        
        # Copy user response
        user_response_dir = os.path.join(dataset_response_user, category, group)
        user_response_path = os.path.join(user_response_dir, user_output_file)
        if os.path.exists(user_response_path):
            try:
                dest_path = os.path.join(group_response_dir, user_output_file)
                shutil.copy2(user_response_path, dest_path)
            except Exception as e:
                print(f"Error copying user response for {category}/{group}: {e}")
                stats["user_response_errors"] += 1
                continue
        
        # Create a verification file with details
        verification_file = os.path.join(group_response_dir, "advanced_verification.txt")
        with open(verification_file, "w", encoding="utf-8") as f:
            f.write(f"Advanced Human Verification for {category}/{group}\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Preferred model from user response: {group_info['preferred_model']}\n")
            f.write(f"Average agreement with preferred model: {group_info['average_agreement']:.2f}\n\n")
            
            # Add preferred model scores
            f.write(f"Preferred model ({group_info['preferred_model']}) scores:\n")
            for i, score in enumerate(group_info['preferred_scores']):
                f.write(f"  Image {i+1}: {score}\n")
            f.write("\n")
            
            # Add model agreement details
            f.write("Agreement with other models:\n")
            for model, agreement in group_info["model_agreements"].items():
                f.write(f"  {model}: {agreement:.2f}\n")
            f.write("\n")
            
            # Add other model scores if available
            if "other_model_scores" in group_info:
                f.write("Scores from other models:\n")
                for model, scores in group_info["other_model_scores"].items():
                    f.write(f"  {model}: {scores}\n")
                f.write("\n")
            
            f.write("-" * 50 + "\n\n")
            f.write("YOUR VERIFICATION\n")
            f.write("-" * 50 + "\n\n")
            f.write("Please provide your expert ranking of these images based on persuasiveness:\n")
            f.write("Rank (comma-separated, e.g. 1,2,3,4): ___________________\n\n")
            f.write("Persuasion scores (comma-separated, 1-100): ___________________\n\n")
            f.write("Comments on why previous rankings may have disagreed: ___________________\n\n")
            f.write("Your reasoning for the ranking (optional): ___________________\n")
        
        stats["successful_copies"] += 1
        print(f"Successfully copied data for {category}/{group}")
    
    # Create a summary report
    summary_file = os.path.join(output_dir, "summary_report.txt")
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("HUMAN VERIFICATION SUBSET SUMMARY\n")
        f.write("================================\n\n")
        f.write(f"Total low agreement groups identified: {len(low_agreement_groups)}\n")
        f.write(f"Successfully copied: {stats['successful_copies']}\n")
        f.write(f"Image copy errors: {stats['image_errors']}\n")
        f.write(f"Response copy errors: {stats['response_errors']}\n")
        f.write(f"User response copy errors: {stats['user_response_errors']}\n\n")
        
        f.write("List of groups by agreement level (ascending):\n")
        sorted_groups = sorted(low_agreement_groups, key=lambda x: x["average_agreement"])
        for idx, group_info in enumerate(sorted_groups, 1):
            preferred_info = f" (Preferred model: {group_info['preferred_model']})"
            f.write(f"{idx}. {group_info['category']}/{group_info['group']} - Agreement: {group_info['average_agreement']:.2f}{preferred_info}\n")
    
    return stats

def create_human_verification_instructions():
    """Create instructions for human verifiers"""
    instructions_file = os.path.join(output_dir, "VERIFICATION_INSTRUCTIONS.md")
    
    with open(instructions_file, "w", encoding="utf-8") as f:
        f.write("# Advanced Human Verification Instructions\n\n")
        
        f.write("## Background\n\n")
        f.write("This dataset contains image groups where AI models had low agreement (< 50%) with the model preferred by previous human evaluators.\n")
        f.write("Your expert judgment is needed to establish a more reliable ground truth and understand why models disagreed.\n\n")
        
        f.write("## Task\n\n")
        f.write("For each group:\n\n")
        f.write("1. Review the images in the corresponding folder under `images/`\n")
        f.write("2. Review all model outputs in the corresponding folder under `responses/`\n")
        f.write("3. Pay special attention to the user's preferred model (noted in the verification file)\n")
        f.write("4. Complete the `advanced_verification.txt` file with your own ranking and scores\n")
        f.write("5. Add comments explaining potential reasons for the disagreement\n\n")
        
        f.write("## Understanding the Problem\n\n")
        f.write("Each group in this dataset was previously evaluated by a human who selected a preferred model.\n")
        f.write("However, other models significantly disagreed with the preferred model's ranking (< 50% agreement).\n")
        f.write("Your task is to provide a more reliable ranking and to understand why these disagreements occurred.\n\n")
        
        f.write("## Ranking Guidelines\n\n")
        f.write("- Focus on persuasiveness for selling the specific product category\n")
        f.write("- Consider factors like image quality, product presentation, and visual appeal\n")
        f.write("- Provide both rankings (ordinal position) and scores (1-100 scale)\n")
        f.write("- For tied images that you consider equally persuasive, assign the same rank\n")
        f.write("- Provide clear reasoning for your ranking decisions\n\n")
        
        f.write("## Analyzing Disagreements\n\n")
        f.write("Consider these potential sources of disagreement:\n\n")
        f.write("1. **Different Focus**: Models might focus on different aspects of persuasiveness\n")
        f.write("2. **Score Scaling**: Models might use different ranges within the 1-100 scale\n")
        f.write("3. **Contextual Understanding**: Models might interpret the product category differently\n")
        f.write("4. **Human vs. AI Perception**: The human evaluator might value different aspects than the models\n\n")
        
        f.write("## Submission\n\n")
        f.write("After completing verification for all assigned groups, submit the entire folder with your completed verification files.\n")
        f.write("Make sure all `advanced_verification.txt` files have your rankings, scores, and reasoning filled in.\n")
    
    # Also create a README at the root
    readme_file = os.path.join(output_dir, "README.md")
    with open(readme_file, "w", encoding="utf-8") as f:
        f.write("# Human Verification Subset\n\n")
        f.write("This subset contains image groups where AI models significantly disagreed with the model preferred by users.\n\n")
        f.write("## Overview\n\n")
        f.write("In the previous evaluation phase, users selected their preferred model for each group.\n")
        f.write("This dataset contains only groups where other models had low agreement (<50%) with the preferred model.\n\n")
        
        f.write("## Contents\n\n")
        f.write("- `images/`: Product images organized by category and group\n")
        f.write("- `responses/`: Model outputs and verification files\n")
        f.write("- `VERIFICATION_INSTRUCTIONS.md`: Detailed instructions for human verifiers\n")
        f.write("- `summary_report.txt`: Overview of the dataset statistics\n")
        f.write("- `disagreement_stats.json`: Detailed metrics on model disagreements\n\n")
        
        f.write("## Quick Start\n\n")
        f.write("1. Read the `VERIFICATION_INSTRUCTIONS.md` file\n")
        f.write("2. For each group in the dataset:\n")
        f.write("   - Review images in the corresponding `images/` subfolder\n")
        f.write("   - Check model outputs in the `responses/` subfolder\n")
        f.write("   - Note which model was preferred by the previous evaluator\n")
        f.write("   - Complete the `advanced_verification.txt` file\n")
        f.write("3. Submit the entire folder with your completed verification files\n")

if __name__ == "__main__":
    print("Starting human verification subset creation...")
    print(f"Data root: {data_root}")
    print(f"Output directory: {output_dir}")
    
    # Identify groups with low agreement
    low_agreement_groups = get_low_agreement_groups()
    
    if low_agreement_groups:
        # Copy data for these groups
        stats = copy_group_data(low_agreement_groups)
        
        # Create instructions for human verifiers
        create_human_verification_instructions()
        
        print(f"\nHuman verification subset created with {stats['successful_copies']} groups")
        print(f"Output directory: {output_dir}")
        
        # Print statistics summary
        print("\nSummary Statistics:")
        print(f"Total groups processed: {len(low_agreement_groups)}")
        print(f"Successfully copied: {stats['successful_copies']}")
        print(f"Image errors: {stats['image_errors']}")
        print(f"Response errors: {stats['response_errors']}")
        print(f"User response errors: {stats['user_response_errors']}")
        
        # Additional analysis
        if stats['successful_copies'] > 0:
            # Calculate percentages
            success_rate = (stats['successful_copies'] / len(low_agreement_groups)) * 100
            print(f"Success rate: {success_rate:.1f}%")
            
            # Calculate average agreement level
            avg_agreement = sum(group["average_agreement"] for group in low_agreement_groups) / len(low_agreement_groups)
            print(f"Average agreement level across subset: {avg_agreement:.4f}")
    else:
        print("No low agreement groups found or error in analysis.")