import os
import torch
import json
import numpy as np
from PIL import Image
import re
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
# Create output directory
output_dir = "./pairwise_evaluation_results_pairwise_2"
os.makedirs(output_dir, exist_ok=True)

## SET PATHS
data_root = "home/debajyoti/paridhi_mtp/product_images_real"
#data_root = "/home/deepg/NAS/Downloads/MTP-2-persuasion/dataset"
dataset_image = os.path.join(data_root, "dataset_image_new")
dataset_response = os.path.join(data_root, "dataset_response_new")
dataset_user_preferred = os.path.join(data_root, "final_data")

# Authenticate with Hugging Face
login(token='hf_xxxxxxxxxxxxxxxx')

# # Define model name
# model_name = "Deb123/qwen2.5-vl-7b-pair-finetuned-private"
#model_name = "vlm_finetuned_full_context_48_2"
model_name = "vlm_finetuned_pairwise"
# model_name = "unsloth/Qwen2.5-VL-7B-Instruct"
# Load model and processor
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = AutoProcessor.from_pretrained(model_name, use_fast=False)
model = AutoModelForImageTextToText.from_pretrained(model_name).to(device)


print(f"Model and processor loaded successfully on {device}!")

def resize_if_needed(img, max_size=384):
    """Resize an image if its dimensions exceed max_size while maintaining aspect ratio."""
    if img.width > max_size or img.height > max_size:
        scaling_factor = max_size / float(max(img.width, img.height))
        new_width = int(img.width * scaling_factor)
        new_height = int(img.height * scaling_factor)
        return img.resize((new_width, new_height), Image.LANCZOS)
    return img


def get_model_ranking(model, processor, images, category):
    """Get model prediction for a set of images."""
    try:
        # Format image paths using file:// prefix like in code 1
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
                        "text": f"You are evaluating images in the '{category}' product category. "
                                f"Rank the images, based on their appeal for selling this '{category}' product. "
                                f"Provide description, and **persuasion score (1-100)** for each image and explain the ranking."
                    }
                ],
            }
        ]
        
        # Process the inputs exactly like in code 1
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
        
        # Write response to a temporary file for extract_text_data
        temp_response_file = os.path.join(output_dir, f"temp_response_{category}_{len(images)}.txt")
        with open(temp_response_file, "w", encoding="utf-8") as f:
            f.write(generated_text)
        
        # Extract information using extract_text_data function
        extracted_info, ranking = extract_text_data(temp_response_file)
        
        # If extraction failed, return None
        if not extracted_info:
            print(f"Failed to extract information from model response for {category}")
            return None
        
        # Process scores and determine ranking
        scores = []
        for i, info in enumerate(extracted_info):
            if "score" in info and info["score"] is not None:
                scores.append((i+1, info["score"]))  # Store (image_index, score)
            else:
                print(f"Missing score for image {i+1} in {category}")
                return None
        
        # Sort by score to get ranking (descending)
        scores.sort(key=lambda x: x[1], reverse=True)
        ranked_images = [idx for idx, _ in scores]
        
        return {
            "extracted_info": extracted_info,
            "ranking": ranking,
            "ranked_images": ranked_images,
            "full_response": generated_text
        }
    except Exception as e:
        print(f"Error in get_model_ranking: {e}")
        return None


def conduct_tournament(model, processor, images, category):
    """Conduct a tournament between images to determine the final ranking."""
    num_images = len(images)
    
    # If only 2 images, direct comparison
    if num_images == 2:
        result = get_model_ranking(model, processor, images, category)
        return result if result else None
    
    # For 3 or 4 images, use tournament structure
    elif num_images == 3:
        # First round: Compare images 0 and 1
        first_round = get_model_ranking(model, processor, images[:2], category)
        if not first_round:
            return None
        
        # Get the winner of the first round
        winner_idx = first_round["ranked_images"][0] - 1  # Adjust for 0-indexing
        
        # Final round: Compare winner with image 2
        final_round = get_model_ranking(model, processor, [images[winner_idx], images[2]], category)
        return final_round
    
    elif num_images == 4:
        # First semifinal: Compare images 0 and 1
        semi1 = get_model_ranking(model, processor, images[:2], category)
        if not semi1:
            return None
        
        # Second semifinal: Compare images 2 and 3
        semi2 = get_model_ranking(model, processor, images[2:], category)
        if not semi2:
            return None
        
        # Get winners from each semifinal
        winner1_idx = semi1["ranked_images"][0] - 1  # Adjust for 0-indexing
        winner2_idx = semi2["ranked_images"][0] - 1 + 2  # Adjust for 0-indexing and offset
        
        # Final: Compare winners
        final_round = get_model_ranking(model, processor, [images[winner1_idx], images[winner2_idx]], category)
        return final_round
    
    else:
        print(f"Unsupported number of images: {num_images}")
        return None

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
                        
                    response_path = os.path.join(dataset_user_preferred, category, group, "user_output.txt")
                    if os.path.exists(response_path):
                        try:
                            extracted_info, ranking = extract_text_data(response_path)
                            if extracted_info:
                                data.append({
                                    "images": images,
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

def evaluate_model_tournament(data, test_size=0.05):
    """Evaluate the model using the tournament approach on a sample of the dataset."""
    # Split dataset into train/val sets
    train_data, test_data = train_test_split(data, test_size=test_size, random_state=45)
    print(f"Evaluation dataset: {len(test_data)} groups out of {len(data)} total groups")
    
    eval_data = []
    rank_matches = 0
    total_cases = 0
    score_diffs = []
    mse_scores = []
    ranking_losses = []
    
    # Process each group in the validation set
    for batch in tqdm(test_data, desc="Processing evaluation groups"):
        category = batch["category"]
        group = batch["group"]
        images = batch["images"]
        
        try:
            # Get ground truth
            ground_truth_info = batch["extracted_info"]
            
            # Skip if no valid ground truth
            if not ground_truth_info or not isinstance(ground_truth_info, list):
                print(f"Invalid ground truth for {category}/{group}")
                continue
            
            # Make sure ground truth data is sorted by image_num
            ground_truth_sorted = sorted(ground_truth_info, key=lambda x: x["image_num"])
            ground_truth_scores = [item["score"] for item in ground_truth_sorted if "score" in item and item["score"] is not None]
            
            if len(ground_truth_scores) == 0:
                print(f"No valid scores in ground truth for {category}/{group}")
                continue
            
            # Convert ground truth scores from NumPy types to Python primitives
            ground_truth_scores = [float(score) for score in ground_truth_scores]
            
            # Track tournament brackets to know which images are compared in the final
            tournament_brackets = []
            
            # Run the tournament
            if len(images) == 2:
                # Direct comparison
                final_match = get_model_ranking(model, processor, images, category)
                if not final_match:
                    print(f"Tournament failed for {category}/{group}")
                    continue
                finalist_indices = [0, 1]  # Original image indices
                
            elif len(images) == 3:
                # First round: Compare images 0 and 1
                first_round = get_model_ranking(model, processor, images[:2], category)
                if not first_round:
                    print(f"First round failed for {category}/{group}")
                    continue
                    
                # Get the winner of the first round
                winner_idx = first_round["ranked_images"][0] - 1  # Convert to 0-indexed
                tournament_brackets.append((0, 1, winner_idx))
                
                # Final round: Compare winner with image 2
                final_match = get_model_ranking(model, processor, [images[winner_idx], images[2]], category)
                if not final_match:
                    print(f"Final round failed for {category}/{group}")
                    continue
                finalist_indices = [winner_idx, 2]  # Original image indices
                
            elif len(images) == 4:
                # First semifinal: Compare images 0 and 1
                semi1 = get_model_ranking(model, processor, images[:2], category)
                if not semi1:
                    print(f"First semifinal failed for {category}/{group}")
                    continue
                    
                # Second semifinal: Compare images 2 and 3
                semi2 = get_model_ranking(model, processor, images[2:], category)
                if not semi2:
                    print(f"Second semifinal failed for {category}/{group}")
                    continue
                    
                # Get winners from each semifinal
                winner1_idx = semi1["ranked_images"][0] - 1  # Convert to 0-indexed
                winner2_idx = semi2["ranked_images"][0] - 1 + 2  # Adjust for indices
                
                tournament_brackets.append((0, 1, winner1_idx))
                tournament_brackets.append((2, 3, winner2_idx))
                
                # Final: Compare winners
                final_match = get_model_ranking(model, processor, [images[winner1_idx], images[winner2_idx]], category)
                if not final_match:
                    print(f"Final round failed for {category}/{group}")
                    continue
                finalist_indices = [winner1_idx, winner2_idx]  # Original image indices
            
            # Convert finalist indices from NumPy types to Python primitives if needed
            finalist_indices = [int(idx) for idx in finalist_indices]
            
            # Get the model's ranking of the final two images
            model_ranking = final_match["ranked_images"]
            
            # Map the model's ranking (1-indexed) back to the original image indices
            model_top_image = finalist_indices[model_ranking[0] - 1] + 1  # Convert to 1-indexed
            
            # Get ground truth top image from all images (1-indexed)
            ground_truth_top_image = int(np.argmax(ground_truth_scores) + 1)
            
            # Calculate rank match (does the model's top pick match ground truth's top pick?)
            rank_match = bool(model_top_image == ground_truth_top_image)
            
            # For MSE, only compare the scores of the two finalists
            # Get the predicted scores for the two finalist images
            predicted_scores_finalists = [
                float(final_match["extracted_info"][i]["score"]) 
                for i in range(len(final_match["extracted_info"]))
                if "score" in final_match["extracted_info"][i] and final_match["extracted_info"][i]["score"] is not None
            ]
            
            # Get the ground truth scores for the two finalist images
            ground_truth_scores_finalists = [ground_truth_scores[idx] for idx in finalist_indices]
            
            # Calculate MSE for finalists only
            if len(predicted_scores_finalists) == len(ground_truth_scores_finalists):
                mse_loss = float(np.mean((np.array(predicted_scores_finalists) - np.array(ground_truth_scores_finalists)) ** 2))
                mse_scores.append(mse_loss)
            else:
                print(f"Finalist score count mismatch in {category}/{group}")
                mse_loss = None
            
            # For ranking loss, use rank correlation measures
            # Get the ground truth ranking of the finalists
            gt_scores_finalists = [ground_truth_scores[idx] for idx in finalist_indices]
            gt_ranking_finalists = np.argsort(np.argsort(-np.array(gt_scores_finalists)))
            
            # Get the model's ranking of the finalists
            model_ranking_finalists = np.argsort(np.argsort(-np.array(predicted_scores_finalists)))
            
            # Calculate normalized ranking loss for finalists
            if len(gt_ranking_finalists) == len(model_ranking_finalists):
                    ranking_loss = float(np.linalg.norm(gt_ranking_finalists - model_ranking_finalists) / len(gt_ranking_finalists))
                    ranking_losses.append(ranking_loss)
            else:
                ranking_loss = None
                        
            # Calculate score difference between top picks
            score_diff = None
            if len(predicted_scores_finalists) >= 1:
                # Find the index of the ground truth top image within the finalists
                if ground_truth_top_image - 1 in finalist_indices:
                    gt_top_idx = finalist_indices.index(ground_truth_top_image - 1)
                    model_top_idx = model_ranking[0] - 1  # Convert to 0-indexed
                    
                    score_diff = float(abs(predicted_scores_finalists[model_top_idx] - ground_truth_scores_finalists[gt_top_idx]))
                    score_diffs.append(score_diff)
            
            # Update counters
            if rank_match:
                rank_matches += 1
            total_cases += 1
            
            # Save detailed result to the evaluation data
            result_data = {
                "category": category,
                "group": group,
                "ground_truth_top_image": int(ground_truth_top_image),
                "model_top_image": int(model_top_image),
                "rank_match": bool(rank_match),
                "ground_truth_scores": [float(score) for score in ground_truth_scores],
                "predicted_scores_finalists": [float(score) for score in predicted_scores_finalists],
                "finalist_indices": [int(idx) for idx in finalist_indices],
                "tournament_brackets": [
                    [int(a), int(b), int(c)] for a, b, c in tournament_brackets
                ] if tournament_brackets else [],
                "mse_loss": float(mse_loss) if mse_loss is not None else None,
                "ranking_loss": float(ranking_loss) if ranking_loss is not None else None,
                "score_diff": float(score_diff) if score_diff is not None else None,
                "model_response": final_match["full_response"]
            }
            
            eval_data.append(result_data)
            
            # Save individual evaluation result - ensure all data is JSON serializable
            with open(os.path.join(output_dir, f"{category}_{group}_metrics.json"), "w") as f:
                json.dump(result_data, f, indent=2)
            
            # Clear GPU memory
            torch.cuda.empty_cache()
            gc.collect()
        
        except Exception as e:
            print(f"Error processing {category}/{group}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    # Calculate final metrics
    rank_accuracy = float(rank_matches / total_cases) if total_cases > 0 else 0
    avg_score_diff = float(np.mean(score_diffs)) if score_diffs else 0
    avg_mse = float(np.mean(mse_scores)) if mse_scores else 0
    avg_ranking_loss = float(np.mean(ranking_losses)) if ranking_losses else 0

    # Save evaluation results
    results = {
        "summary": {
            "total_cases": int(total_cases),
            "rank_matches": int(rank_matches),
            "rank_accuracy": float(rank_accuracy),
            "average_score_diff": float(avg_score_diff),
            "average_mse": float(avg_mse),
            "average_ranking_loss": float(avg_ranking_loss)
        },
        "detailed_results": eval_data
    }

    with open(os.path.join(output_dir, "evaluation_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Also save raw outputs
    with open(os.path.join(output_dir, "raw_outputs.txt"), "w", encoding="utf-8") as f:
        for item in eval_data:
            f.write(f"\n\n{'='*50}\n")
            f.write(f"CATEGORY: {item['category']} | GROUP: {item['group']}\n")
            f.write(f"{'='*50}\n\n")
            f.write(item['model_response'])
            f.write(f"\n{'='*50}\n")

    print(f"\nEvaluation complete. Results saved to {os.path.join(output_dir, 'evaluation_results.json')}")
    print(f"Total groups evaluated: {total_cases}")
    print(f"Rank Accuracy: {rank_accuracy:.4f}")
    print(f"Mean Score Difference: {avg_score_diff:.4f}")
    print(f"Mean Squared Error (MSE): {avg_mse:.4f}")
    print(f"Ranking Loss (Normalized): {avg_ranking_loss:.4f}")

    return results

if __name__ == "__main__":
    data = load_dataset()
    evaluate_model_tournament(data)
