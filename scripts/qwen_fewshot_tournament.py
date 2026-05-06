import os
import torch
import json
import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from tqdm import tqdm
import gc
from qwen_vl_utils import process_vision_info
from simple_data_preprocess import extract_text_data
from huggingface_hub import login
from sklearn.model_selection import train_test_split
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
data_root = "/home/debajyoti/paridhi_mtp/MTP-2-persuasion (Datasets, Results, PPTs)/MTP-2-persuasion/dataset"
#data_root = "/home/deepg/NAS/Downloads/MTP-2-persuasion/dataset"
dataset_image = os.path.join(data_root, "dataset_image")
dataset_response = os.path.join(data_root, "final_data")
fewshot_folder = "/home/debajyoti/paridhi_mtp/MTP-2-persuasion (Datasets, Results, PPTs)/MTP-2-persuasion/fewshot_for_deep/images"
#fewshot_folder = "/home/deepg/NAS/Downloads/MTP-2-persuasion/fewshot_for_deep/images"  # Few-shot examples folder
score_folder = "/home/debajyoti/paridhi_mtp/MTP-2-persuasion (Datasets, Results, PPTs)/MTP-2-persuasion/fewshot_for_deep/scores"
#score_folder = "/home/deepg/NAS/Downloads/MTP-2-persuasion/fewshot_for_deep/scores"    # Few-shot scores folder
metrics_output_dir = os.path.join(data_root, "dataset_response_for_tournament")
debug_output_dir = os.path.join(data_root, "tournament_detailed_logs_fewshot")
os.makedirs(metrics_output_dir, exist_ok=True)
os.makedirs(debug_output_dir, exist_ok=True)

# Authenticate with Hugging Face
login(token='hf_xxxxxxxxx')

# Define model name
model_name = "qwen/Qwen2.5-VL-7B-Instruct"

# Load model and processor
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = AutoProcessor.from_pretrained(model_name, use_fast=False)
model = AutoModelForImageTextToText.from_pretrained(model_name).to(device)

print(f"Model and processor loaded successfully on {device}!")

def get_one_shot_example(category):
    """Fetches the first group in `fewshot/` and its corresponding score from `scores/`."""
    temp_category_path = os.path.join(fewshot_folder, category)
    score_category_path = os.path.join(score_folder, category)

    if not os.path.isdir(temp_category_path) or not os.path.isdir(score_category_path):
        return None

    groups = sorted(os.listdir(temp_category_path))  # Get all groups in category
    if not groups:
        return None

    first_group = groups[0]  # Select the first group
    group_path = os.path.join(temp_category_path, first_group)
    score_path = os.path.join(score_category_path, first_group, "score.txt")

    if not os.path.isdir(group_path) or not os.path.exists(score_path):
        return None

    # First sort filenames, then create dictionaries
    image_files = sorted([
        img for img in os.listdir(group_path)
        if img.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
    ])
    
    images = [os.path.join(group_path, img) for img in image_files]

    # Read the corresponding `score.txt`
    with open(score_path, "r", encoding="utf-8") as f:
        score_text = f.read().strip()

    print(f"One-shot example from '{first_group}' in '{category}' with {len(images)} images.")

    return {"group": first_group, "images": images, "score": score_text}

def extract_fewshot_output(file_path):
    """
    Extract information from model outputs that include few-shot examples.
    This function separates the few-shot examples from the actual model output.
    
    Args:
        file_path: Path to the file containing the model output
    
    Returns:
        tuple: (extracted_info, ranking_text)
    """
    with open(file_path, "r", encoding="utf-8") as f:
        raw_text = f.read().strip()
    
    # Split the output to isolate the model's actual response
    # Find where the prompt ends and the actual evaluation starts
    separator_patterns = [
        "Now, evaluate these two images",
        "Now, evaluate new images",
        "Compare the images and determine",
        "assistant"
    ]
    print("[INFO] Processing file: ALL good")
    split_position = -1
    for pattern in separator_patterns:
        pos = raw_text.find(pattern)
        if pos > -1:
            split_position = pos
            break
    print(split_position)
    # If we found a separator, use only the text after it
    if split_position > -1:
        # Find the end of the instruction sentence
        instruction_end = raw_text.find("\n", split_position)
        if instruction_end > -1:
            # Take only the text after the instruction
            actual_output = raw_text[instruction_end:].strip()
        else:
            # If we can't find the end of the instruction, use a best guess
            actual_output = raw_text[split_position + 50:].strip()
            
        print(f"Separated few-shot examples from actual output. Output length: {len(actual_output)}")
    else:
        # If no separator found, use the original text but log a warning
        actual_output = raw_text
        print("WARNING: Could not identify few-shot example separator. Results may be unreliable.")
    
    # Write the separated output to a temporary file
    temp_file = f"{file_path}_processed"
    with open(temp_file, "w", encoding="utf-8") as f:
        f.write(actual_output)
    
    # Use the existing extract_text_data on the processed file
    from MTP_new_preprocess import extract_text_data
    return extract_text_data(temp_file)

# Update the get_pairwise_ranking function to use the new extraction function
def get_pairwise_ranking(model, processor, image_pair, category, fewshot_example=None, round_name="", debug_info=None):
    """Get model prediction for a strictly pairwise comparison of images with detailed debugging."""
    # Initialize round info for the debug log
    round_info = {
        "round_name": round_name,
        "images_compared": [os.path.basename(img) for img in image_pair],
        "images_indices": list(range(len(image_pair))),  # Will always be [0, 1]
        "status": "processing",
        "error": None
    }
    
    if debug_info is not None:
        debug_info["rounds"].append(round_info)
    
    try:
        # Format image paths using file:// prefix - ONLY TWO IMAGES
        formatted_images = []
        for img_path in image_pair:
            formatted_images.append({"type": "image", "image": f"file://{img_path}"})
        
        # Prepare content list for the message
        content = []
        
        # Add few-shot example if provided
        if fewshot_example:
            # Add example header
            content.append({
                "type": "text",
                "text": f"Example from group '{fewshot_example['group']}' in '{category}' category:"
            })
            
            # Add example images
            for img_path in fewshot_example['images']:
                content.append({"type": "image", "image": f"file://{img_path}"})
            
            # Add example explanation
            content.append({
                "type": "text",
                "text": f"Example persuasion score and explanation for group {fewshot_example['group']}:\n{fewshot_example['score']}"
            })
            
            # Add separator and instruction for new images - this will be detected later
            content.append({
                "type": "text",
                "text": f"\nNow, evaluate these two images in the '{category}' product category. Compare the images and determine which one is better for selling this '{category}' product. Provide description, and **persuasion score (1-100)** for each image and explain your ranking."
            })
        else:
            # If no few-shot example, use direct instruction
            content.append({
                "type": "text",
                "text": f"You are evaluating these two images in the '{category}' product category. "
                        f"Compare the images and determine which one is better for selling this '{category}' product. "
                        f"Provide description, and **persuasion score (1-100)** for each image and explain your ranking."
            })
        
        # Add the two images to evaluate
        content.extend(formatted_images)
        
        # Create user message
        messages = [{"role": "user", "content": content}]
        
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
        model.eval()
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=800)

        # Decode output
        generated_text = processor.batch_decode(output, skip_special_tokens=True)[0]
        generated_text = generated_text.split("assistant\n", 1)[-1].strip()
        # Create temporary file for extract_text_data
        temp_dir = os.path.join(debug_output_dir, "temp")
        os.makedirs(temp_dir, exist_ok=True)
        temp_response_file = os.path.join(temp_dir, f"temp_response_{category}_{round_name}.txt")
        with open(temp_response_file, "w", encoding="utf-8") as f:
            f.write(generated_text)
        
        print(1111111111111111111111111)
        # Use our new extraction function that handles few-shot examples
        extracted_info, ranking = extract_fewshot_output(temp_response_file)
        
        # Log the number of extracted images for debugging
        print(f"Extracted {len(extracted_info)} image entries")
        
        # If extraction failed, return None
        if not extracted_info:
            error_msg = f"Failed to extract information from model response for {category} ({round_name})"
            print(f"{error_msg}")
            if debug_info is not None:
                round_info["status"] = "failed"
                round_info["error"] = error_msg
            return None
        
        # Verify we have exactly 2 scores (for the image pair we're comparing)
        if len(extracted_info) != 2:
            error_msg = f"Expected exactly 2 scores but got {len(extracted_info)} for {category} ({round_name})"
            print(f"{error_msg}")
            
            # Try to filter out examples if we have more than 2 scores
            if len(extracted_info) > 2 and fewshot_example:
                print("Attempting to filter out example scores...")
                # Take only the last 2 entries (assuming they're the actual comparison)
                extracted_info = extracted_info[-2:]
                print(f"After filtering: {len(extracted_info)} entries")
                
                # Still not 2 entries? Return None
                if len(extracted_info) != 2:
                    if debug_info is not None:
                        round_info["status"] = "failed"
                        round_info["error"] = error_msg
                    return None
            else:
                if debug_info is not None:
                    round_info["status"] = "failed"
                    round_info["error"] = error_msg
                return None
            
        scores = []
        for i, info in enumerate(extracted_info):
            if "score" in info and info["score"] is not None:
                scores.append((i+1, info["score"]))  # Store (image_index, score)
            else:
                error_msg = f"Missing score for image {i+1} in {category} ({round_name})"
                print(f"{error_msg}")
                if debug_info is not None:
                    round_info["status"] = "failed"
                    round_info["error"] = error_msg
                return None
        
        # Sort by score to get ranking (descending)
        scores.sort(key=lambda x: x[1], reverse=True)
        ranked_images = [idx for idx, _ in scores]
        raw_scores = [score for _, score in scores]
        
        # Update round info in debug log
        if debug_info is not None:
            round_info["status"] = "success"
            round_info["model_scores"] = raw_scores
            round_info["ranked_images"] = ranked_images
            round_info["ranking_explanation"] = ranking
            round_info["full_response"] = generated_text
            
            # Identify the winner (1-indexed in this comparison)
            round_info["winner"] = ranked_images[0]
            
        return {
            "extracted_info": extracted_info,
            "ranking": ranking,
            "ranked_images": ranked_images,
            "scores": raw_scores,
            "full_response": generated_text
        }
    except Exception as e:
        error_msg = f"Error in get_pairwise_ranking for {category} ({round_name}): {str(e)}"
        print(error_msg)
        if debug_info is not None:
            round_info["status"] = "failed"
            round_info["error"] = error_msg
        return None

def conduct_tournament(model, processor, images, category, group, debug_info, fewshot_example=None):
    """Conduct a tournament between images with comprehensive debugging using strict pairwise comparisons."""
    num_images = len(images)
    debug_info["images"] = [os.path.basename(img) for img in images]
    debug_info["tournament_structure"] = f"{num_images}-image tournament"
    debug_info["rounds"] = []
    
    # Record the original indices for tracking
    original_indices = list(range(num_images))
    
    # If only 2 images, direct comparison
    if num_images == 2:
        debug_info["tournament_type"] = "direct_comparison"
        result = get_pairwise_ranking(model, processor, images, category, 
                                  fewshot_example, f"{group}_direct_comparison", debug_info)
        if not result:
            debug_info["status"] = "failed"
            debug_info["error"] = "Direct comparison failed"
            return None, [], original_indices
        
        debug_info["status"] = "success"
        # The finalists are the original images (0 and 1)
        return result, [], original_indices
    
    # For 3 images, use tournament structure
    elif num_images == 3:
        debug_info["tournament_type"] = "three_image_tournament"
        tournament_structure = []
        
        # First round: Compare images 0 and 1 (strictly pairwise)
        first_round = get_pairwise_ranking(model, processor, images[:2], category, 
                                       fewshot_example, f"{group}_round1_pair01", debug_info)
        if not first_round:
            debug_info["status"] = "failed"
            debug_info["error"] = "First round comparison failed"
            return None, [], original_indices
        
        # Get the winner of the first round (0-indexed in the original list)
        winner_idx = first_round["ranked_images"][0] - 1  # Adjust for 0-indexing
        tournament_structure.append((0, 1, winner_idx))
        
        # For debug: which original image won the first round
        debug_info["first_round_winner"] = {
            "original_index": winner_idx,
            "filename": os.path.basename(images[winner_idx])
        }
        
        # Create finalist images list for the final round (strictly pairwise)
        finalist_images = [images[winner_idx], images[2]]
        finalist_indices = [winner_idx, 2]  # Original indices
        
        # Final round: Compare winner with image 2 (strictly pairwise)
        final_round = get_pairwise_ranking(model, processor, finalist_images, category, 
                                       fewshot_example, f"{group}_final_round", debug_info)
        if not final_round:
            debug_info["status"] = "failed"
            debug_info["error"] = "Final round comparison failed"
            return None, tournament_structure, original_indices
            
        # For debug: final winner
        final_winner_local_idx = final_round["ranked_images"][0] - 1  # 0 or 1 in this pair
        final_winner_original_idx = finalist_indices[final_winner_local_idx]  # Map back to original index
        
        debug_info["final_winner"] = {
            "local_index": final_winner_local_idx,
            "original_index": final_winner_original_idx,
            "filename": os.path.basename(images[final_winner_original_idx])
        }
        
        debug_info["status"] = "success"
        return final_round, tournament_structure, finalist_indices
    
    elif num_images == 4:
        debug_info["tournament_type"] = "four_image_tournament"
        tournament_structure = []
        
        # First semifinal: Compare images 0 and 1 (strictly pairwise)
        semi1 = get_pairwise_ranking(model, processor, images[:2], category, 
                                  fewshot_example, f"{group}_semifinal1_pair01", debug_info)
        if not semi1:
            debug_info["status"] = "failed"
            debug_info["error"] = "First semifinal failed"
            return None, [], original_indices
        
        # Get the winner of the first semifinal (0-indexed in the original list)
        winner1_idx = semi1["ranked_images"][0] - 1  # Adjust for 0-indexing
        tournament_structure.append((0, 1, winner1_idx))
        
        # For debug: which original image won the first semifinal
        debug_info["semifinal1_winner"] = {
            "original_index": winner1_idx,
            "filename": os.path.basename(images[winner1_idx])
        }
        
        # Second semifinal: Compare images 2 and 3 (strictly pairwise)
        semi2 = get_pairwise_ranking(model, processor, images[2:], category, 
                                 fewshot_example, f"{group}_semifinal2_pair23", debug_info)
        if not semi2:
            debug_info["status"] = "failed"
            debug_info["error"] = "Second semifinal failed"
            return None, tournament_structure, original_indices
        
        # Get the winner of the second semifinal
        # The indices in semi2["ranked_images"] are 1-indexed, referring to indices in the slice [2:]
        # So we need to adjust by adding 2 to get the original indices
        winner2_idx_local = semi2["ranked_images"][0] - 1  # 0 or 1 in this pair (0-indexed)
        winner2_idx = winner2_idx_local + 2  # Map back to original index
        tournament_structure.append((2, 3, winner2_idx))
        
        # For debug: which original image won the second semifinal
        debug_info["semifinal2_winner"] = {
            "local_index": winner2_idx_local,
            "original_index": winner2_idx,
            "filename": os.path.basename(images[winner2_idx])
        }
        
        # Create finalist images list for the final round (strictly pairwise)
        finalist_images = [images[winner1_idx], images[winner2_idx]]
        finalist_indices = [winner1_idx, winner2_idx]  # Original indices
        
        # Final: Compare winners (strictly pairwise)
        final_round = get_pairwise_ranking(model, processor, finalist_images, category, 
                                       fewshot_example, f"{group}_final_round", debug_info)
        if not final_round:
            debug_info["status"] = "failed"
            debug_info["error"] = "Final round failed"
            return None, tournament_structure, finalist_indices
        
        # For debug: final winner
        final_winner_local_idx = final_round["ranked_images"][0] - 1  # 0 or 1 in this pair
        final_winner_original_idx = finalist_indices[final_winner_local_idx]  # Map back to original index
        
        debug_info["final_winner"] = {
            "local_index": final_winner_local_idx,
            "original_index": final_winner_original_idx,
            "filename": os.path.basename(images[final_winner_original_idx])
        }
        
        debug_info["finalist_indices"] = finalist_indices
        debug_info["finalist_filenames"] = [os.path.basename(images[idx]) for idx in finalist_indices]
        debug_info["status"] = "success"
        return final_round, tournament_structure, finalist_indices
    
    else:
        debug_info["status"] = "failed"
        debug_info["error"] = f"Unsupported number of images: {num_images}"
        print(f"Unsupported number of images: {num_images}")
        return None, [], original_indices

def load_dataset():
    """Load the dataset and prepare it for evaluation."""
    data = []
    MAX_IMAGES = 4  # Consistent with training script

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
                        
                    response_path = os.path.join(dataset_response, category, group, "user_output.txt")
                    if os.path.exists(response_path):
                        try:
                            extracted_info, ranking = extract_text_data(response_path)
                            if extracted_info:
                                # Also store image filenames for debug output
                                image_filenames = [os.path.basename(img) for img in images]
                                
                                data.append({
                                    "images": images,
                                    "image_filenames": image_filenames,
                                    "extracted_info": extracted_info,
                                    "ranking": ranking,
                                    "category": category,
                                    "group": group
                                })
                        except Exception as e:
                            print(f"Error processing {category}/{group}: {str(e)}")
                            continue

    print(f"Dataset loaded with {len(data)} groups")
    return data

def evaluate_tournament_with_detailed_logging(data):
    """Evaluate model using tournament approach with comprehensive debugging."""
    
    # Split dataset into train/val sets - using only 5% for validation
    train_data, test_data = train_test_split(data, test_size=0.05, random_state=50)
    print(f"Evaluation dataset: {len(test_data)} groups out of {len(data)} total groups")
    
    # Summary statistics
    stats = {
        "total": len(test_data),
        "success": 0,
        "failure": 0,
        "metrics": {
            "rank_accuracy": [],
            "mse": [],
            "ranking_loss": []
        }
    }
    
    # Process each group in the evaluation set
    for batch in tqdm(test_data, desc="Processing evaluation groups"):
        category = batch["category"]
        group = batch["group"]
        images = batch["images"]
        image_filenames = batch["image_filenames"]
        
        # Get few-shot example for this category
        fewshot_example = get_one_shot_example(category)
        if not fewshot_example:
            print(f"No few-shot example found for {category}/{group}. Using zero-shot approach.")
        
        # Create directory for this category/group
        group_dir = os.path.join(metrics_output_dir, category, group)
        os.makedirs(group_dir, exist_ok=True)
        
        # Output file paths
        metrics_file = os.path.join(group_dir, "output_qwen_fewshot.txt")
        debug_file = os.path.join(debug_output_dir, f"{category}_{group}_detailed.json")
        debug_summary_file = os.path.join(debug_output_dir, f"{category}_{group}_summary.txt")
        
        # Initialize debug info dictionary
        debug_info = {
            "category": category,
            "group": group,
            "image_filenames": image_filenames,
            "timestamp": np.datetime_as_string(np.datetime64('now')),
            "model": "qwen_fewshot",
            "fewshot_example": {
                "available": fewshot_example is not None,
                "group": fewshot_example["group"] if fewshot_example else None,
                "num_images": len(fewshot_example["images"]) if fewshot_example else 0
            }
        }
        
        try:
            # Get ground truth info
            ground_truth_info = batch["extracted_info"]
            
            # Skip if no valid ground truth
            if not ground_truth_info or not isinstance(ground_truth_info, list):
                with open(metrics_file, "w", encoding="utf-8") as f:
                    f.write("ERROR: Invalid ground truth data")
                debug_info["status"] = "failed"
                debug_info["error"] = "Invalid ground truth data"
                
                # Save debug info
                with open(debug_file, "w", encoding="utf-8") as f:
                    json.dump(debug_info, f, indent=2)
                
                stats["failure"] += 1
                continue
            
            # Get ground truth scores
            ground_truth_sorted = sorted(ground_truth_info, key=lambda x: x["image_num"])
            ground_truth_scores = [item.get("score", 0) for item in ground_truth_sorted]
            
            if not all(score is not None for score in ground_truth_scores):
                with open(metrics_file, "w", encoding="utf-8") as f:
                    f.write("ERROR: Missing scores in ground truth data")
                debug_info["status"] = "failed"
                debug_info["error"] = "Missing scores in ground truth data"
                
                # Save debug info
                with open(debug_file, "w", encoding="utf-8") as f:
                    json.dump(debug_info, f, indent=2)
                
                stats["failure"] += 1
                continue
            
            # Record ground truth info in debug_info
            debug_info["ground_truth"] = {
                "scores": ground_truth_scores,
                "top_image_idx": int(np.argmax(ground_truth_scores)),  # 0-indexed
                "top_image_filename": image_filenames[int(np.argmax(ground_truth_scores))]
            }
            
            # Run the tournament for this group
            final_match, tournament_brackets, finalist_indices = conduct_tournament(
                model, processor, images, category, group, debug_info, fewshot_example
            )
            
            if not final_match:
                with open(metrics_file, "w", encoding="utf-8") as f:
                    f.write("ERROR: Tournament failed to complete")
                
                # Save debug info
                with open(debug_file, "w", encoding="utf-8") as f:
                    json.dump(debug_info, f, indent=2)
                
                stats["failure"] += 1
                continue
            
            # Get the model's ranking of the final two images
            model_ranking = final_match["ranked_images"]
            
            # Map the model's ranking (1-indexed) back to the original image indices
            if len(model_ranking) > 0:
                model_top_index = model_ranking[0] - 1  # Convert to 0-indexed within finalists
                if model_top_index < len(finalist_indices):
                    model_top_image = finalist_indices[model_top_index]  # Original 0-indexed position
                else:
                    debug_info["status"] = "failed"
                    debug_info["error"] = f"Invalid model top index: {model_top_index}, finalist_indices: {finalist_indices}"
                    continue
            else:
                debug_info["status"] = "failed"
                debug_info["error"] = "No ranked images in final match"
                continue
            
            # Get ground truth top image (0-indexed)
            ground_truth_top_image = np.argmax(ground_truth_scores)
            
            debug_info["model_prediction"] = {
                "top_image_idx": int(model_top_image),  # 0-indexed
                "top_image_filename": image_filenames[int(model_top_image)]
            }
            
            # Calculate rank match (does the model's top pick match ground truth's top pick?)
            rank_accuracy = 1.0 if model_top_image == ground_truth_top_image else 0.0
            
            # Calculate MSE for the finalist scores
            # Get the predicted scores for the two finalist images
            predicted_scores_finalists = [
                float(final_match["extracted_info"][i]["score"]) 
                for i in range(len(final_match["extracted_info"]))
                if "score" in final_match["extracted_info"][i] and final_match["extracted_info"][i]["score"] is not None
            ]
            
            # Get the ground truth scores for the two finalist images
            ground_truth_scores_finalists = [ground_truth_scores[idx] for idx in finalist_indices]
            
            debug_info["finalist_metrics"] = {
                "finalist_original_indices": [int(idx) for idx in finalist_indices],
                "finalist_filenames": [image_filenames[idx] for idx in finalist_indices],
                "model_scores": predicted_scores_finalists,
                "ground_truth_scores": ground_truth_scores_finalists
            }
            
            # Calculate MSE
            if len(predicted_scores_finalists) == len(ground_truth_scores_finalists):
                mse = float(np.mean((np.array(predicted_scores_finalists) - np.array(ground_truth_scores_finalists)) ** 2))
            else:
                with open(metrics_file, "w", encoding="utf-8") as f:
                    f.write("ERROR: Finalist score count mismatch")
                debug_info["status"] = "failed"
                debug_info["error"] = "Finalist score count mismatch"
                continue
            
            # For ranking loss, use rank correlation measures
            # Get the ground truth ranking of the finalists
            gt_scores_finalists = [ground_truth_scores[idx] for idx in finalist_indices]
            gt_ranking_finalists = np.argsort(np.argsort(-np.array(gt_scores_finalists)))
            
            # Get the model's ranking of the finalists
            model_ranking_finalists = np.argsort(np.argsort(-np.array(predicted_scores_finalists)))
            
            # Calculate normalized ranking loss for finalists
            if len(gt_ranking_finalists) == len(model_ranking_finalists):
                ranking_loss = float(np.linalg.norm(gt_ranking_finalists - model_ranking_finalists) / len(gt_ranking_finalists))
            else:
                with open(metrics_file, "w", encoding="utf-8") as f:
                    f.write("ERROR: Finalist ranking length mismatch")
                debug_info["status"] = "failed"
                debug_info["error"] = "Finalist ranking length mismatch"
                continue
            
            # Record metrics in debug_info
            debug_info["metrics"] = {
                "rank_accuracy": float(rank_accuracy),
                "mse": float(mse),
                "ranking_loss": float(ranking_loss)
            }
            
            # Save metrics to the output file
            with open(metrics_file, "w", encoding="utf-8") as f:
                f.write(f"rank_accuracy: {rank_accuracy:.4f}\n")
                f.write(f"mse: {mse:.4f}\n")
                f.write(f"ranking_loss: {ranking_loss:.4f}\n")
            
            # Save full debug info as JSON
            with open(debug_file, "w", encoding="utf-8") as f:
                json.dump(debug_info, f, indent=2)
            
            # Write human-readable summary
            with open(debug_summary_file, "w", encoding="utf-8") as f:
                f.write(f"TOURNAMENT SUMMARY FOR {category}/{group}\n")
                f.write("=" * 60 + "\n\n")
                
                f.write(f"Category: {category}\n")
                f.write(f"Group: {group}\n")
                f.write(f"Number of images: {len(images)}\n")
                f.write(f"Image filenames: {', '.join(image_filenames)}\n\n")
                
                if fewshot_example:
                    f.write(f"Few-shot example: Group '{fewshot_example['group']}' from category '{category}'\n")
                    f.write(f"Few-shot example images: {len(fewshot_example['images'])}\n\n")
                else:
                    f.write("No few-shot example available. Used zero-shot approach.\n\n")
                
                f.write("GROUND TRUTH\n")
                f.write("-" * 40 + "\n")
                for i, (filename, score) in enumerate(zip(image_filenames, ground_truth_scores)):
                    f.write(f"Image {i}: {filename} - Score: {score}\n")
                f.write(f"\nGround truth top image: Image {ground_truth_top_image} ({image_filenames[ground_truth_top_image]})\n\n")
                
                f.write("TOURNAMENT DETAILS\n")
                f.write("-" * 40 + "\n")
                f.write(f"Tournament type: {debug_info['tournament_type']}\n\n")
                
                if len(images) == 2:
                    f.write("Direct comparison of 2 images\n")
                    if "rounds" in debug_info and len(debug_info["rounds"]) > 0:
                        round_info = debug_info["rounds"][0]
                        f.write(f"Scores: {round_info.get('model_scores', [])}\n")
                        f.write(f"Winner: Image {round_info.get('winner', '?')-1} ({image_filenames[round_info.get('winner', 1)-1]})\n\n")
                
                elif len(images) == 3:
                    f.write("3-image tournament (strictly pairwise comparisons):\n")
                    f.write("First round: Images 0 vs 1\n")
                    if "first_round_winner" in debug_info:
                        winner_idx = debug_info["first_round_winner"]["original_index"]
                        f.write(f"First round winner: Image {winner_idx} ({image_filenames[winner_idx]})\n")
                    
                    f.write("\nFinal round: First round winner vs Image 2\n")
                    if "final_winner" in debug_info:
                        winner_idx = debug_info["final_winner"]["original_index"]
                        f.write(f"Final winner: Image {winner_idx} ({image_filenames[winner_idx]})\n\n")
                
                elif len(images) == 4:
                    f.write("4-image tournament (strictly pairwise comparisons):\n")
                    f.write("First semifinal: Images 0 vs 1\n")
                    if "semifinal1_winner" in debug_info:
                        winner_idx = debug_info["semifinal1_winner"]["original_index"]
                        f.write(f"First semifinal winner: Image {winner_idx} ({image_filenames[winner_idx]})\n")
                    
                    f.write("\nSecond semifinal: Images 2 vs 3\n")
                    if "semifinal2_winner" in debug_info:
                        winner_idx = debug_info["semifinal2_winner"]["original_index"]
                        f.write(f"Second semifinal winner: Image {winner_idx} ({image_filenames[winner_idx]})\n")
                    
                    f.write("\nFinal round: First semifinal winner vs Second semifinal winner\n")
                    if "final_winner" in debug_info:
                        winner_idx = debug_info["final_winner"]["original_index"]
                        f.write(f"Final winner: Image {winner_idx} ({image_filenames[winner_idx]})\n\n")
                
                f.write("FINALISTS\n")
                f.write("-" * 40 + "\n")
                for i, idx in enumerate(finalist_indices):
                    f.write(f"Finalist {i+1}: Image {idx} ({image_filenames[idx]})\n")
                f.write(f"\nGround truth scores for finalists: {ground_truth_scores_finalists}\n")
                f.write(f"Model scores for finalists: {predicted_scores_finalists}\n\n")
                
                f.write("METRICS\n")
                f.write("-" * 40 + "\n")
                f.write(f"Rank Accuracy: {rank_accuracy:.4f}\n")
                if rank_accuracy == 1.0:
                    f.write("  ✓ Model's top pick matches ground truth's top pick\n")
                else:
                    f.write(f"  ✗ Model's top pick (Image {model_top_image}) doesn't match ground truth's top pick (Image {ground_truth_top_image})\n")
                
                f.write(f"MSE: {mse:.4f}\n")
                f.write(f"Ranking Loss: {ranking_loss:.4f}\n")
            
            # Update stats
            stats["success"] += 1
            stats["metrics"]["rank_accuracy"].append(rank_accuracy)
            stats["metrics"]["mse"].append(mse)
            stats["metrics"]["ranking_loss"].append(ranking_loss)
            
            # Clear GPU memory
            torch.cuda.empty_cache()
            gc.collect()
            
        except Exception as e:
            print(f"Error processing {category}/{group}: {str(e)}")
            with open(metrics_file, "w", encoding="utf-8") as f:
                f.write(f"ERROR: {str(e)}")
            
            debug_info["status"] = "failed"
            debug_info["error"] = str(e)
            
            # Save debug info
            with open(debug_file, "w", encoding="utf-8") as f:
                json.dump(debug_info, f, indent=2)
            
            stats["failure"] += 1
            continue
    
    # Calculate average metrics
    if stats["success"] > 0:
        avg_rank_accuracy = np.mean(stats["metrics"]["rank_accuracy"])
        avg_mse = np.mean(stats["metrics"]["mse"])
        avg_ranking_loss = np.mean(stats["metrics"]["ranking_loss"])
    
        # Write summary file
        summary_file = os.path.join(debug_output_dir, "qwen_fewshot_summary.txt")
        with open(summary_file, "w", encoding="utf-8") as f:
            f.write("QWEN FEW-SHOT TOURNAMENT EVALUATION SUMMARY\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Total groups in evaluation set: {stats['total']}\n")
            f.write(f"Successful evaluations: {stats['success']} ({stats['success']/stats['total']*100:.2f}%)\n")
            f.write(f"Failed evaluations: {stats['failure']} ({stats['failure']/stats['total']*100:.2f}%)\n\n")
            
            f.write("Average Metrics:\n")
            f.write(f"Rank Accuracy: {avg_rank_accuracy:.4f}\n")
            f.write(f"Mean Squared Error: {avg_mse:.4f}\n")
            f.write(f"Ranking Loss: {avg_ranking_loss:.4f}\n\n")
            
            f.write("Explanation of Metrics:\n")
            f.write("- Rank Accuracy: Measures if the model's top pick matches ground truth's top pick\n")
            f.write("- MSE: Mean Squared Error between model's scores and ground truth scores for finalist images\n")
            f.write("- Ranking Loss: Normalized difference between model's ranking and ground truth ranking of finalist images\n")
            
        # Save all stats as JSON for further analysis
        complete_stats_file = os.path.join(debug_output_dir, "complete_stats.json")
        with open(complete_stats_file, "w", encoding="utf-8") as f:
            json.dump({
                "summary": {
                    "total": int(stats["total"]),
                    "success": int(stats["success"]),
                    "failure": int(stats["failure"]),
                    "avg_rank_accuracy": float(avg_rank_accuracy),
                    "avg_mse": float(avg_mse),
                    "avg_ranking_loss": float(avg_ranking_loss)
                },
                "metrics": {
                    "rank_accuracy": [float(x) for x in stats["metrics"]["rank_accuracy"]],
                    "mse": [float(x) for x in stats["metrics"]["mse"]],
                    "ranking_loss": [float(x) for x in stats["metrics"]["ranking_loss"]]
                }
            }, f, indent=2)
        
        print(f"\nEvaluation complete.")
        print(f"Metrics saved to: {metrics_output_dir}")
        print(f"Detailed debug logs saved to: {debug_output_dir}")
        print(f"Summary file: {summary_file}")
        print(f"Complete stats: {complete_stats_file}")
        
        print(f"\nOverall metrics (successful evaluations: {stats['success']}):")
        print(f"Rank Accuracy: {avg_rank_accuracy:.4f}")
        print(f"Mean Squared Error: {avg_mse:.4f}")
        print(f"Ranking Loss: {avg_ranking_loss:.4f}")
    else:
        print("No successful evaluations.")

if __name__ == "__main__":
    eval_data = load_dataset()
    evaluate_tournament_with_detailed_logging(eval_data)