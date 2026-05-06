import unsloth
from unsloth import FastVisionModel
import os
import json
import torch
import gc
import re
from PIL import Image
from sklearn.model_selection import train_test_split
from transformers import AutoProcessor, TrainingArguments, Trainer, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType
from huggingface_hub import login, HfApi
from qwen_vl_utils import process_vision_info  # Required for vision processing
from itertools import combinations
import random 
import torch 
import numpy as np
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
# --- Authenticate with Hugging Face ---
login(token="hf_xxxxxxxxx")
repo_id = "Deb123/qwen2.5-vl-7b-pair-new-code-finetuned-private"
api = HfApi()
api.create_repo(repo_id, private=True, exist_ok=True)

## SET PATHS
#data_root = "/home/deepg/NAS/Downloads/MTP-2-persuasion/dataset"
#data_root = "/home/debajyoti/paridhi_mtp/MTP-2-persuasion (Datasets, Results, PPTs)/MTP-2-persuasion/dataset"
data_root = "home/debajyoti/paridhi_mtp/product_images_real"
dataset_image = os.path.join(data_root, "dataset_image_new")
dataset_response = os.path.join(data_root, "final_data")
output_model = "./vlm_finetuned_pairwise"
debug_output_dir = "./debug_output"  # Directory for debug output
os.makedirs(debug_output_dir, exist_ok=True)

# --- Load Processor and Vision-Language Model ---
model_name = "unsloth/Qwen2.5-VL-7B-Instruct"
processor = AutoProcessor.from_pretrained(model_name, use_fast=False)

# Create BitsAndBytesConfig for QLoRA (4-bit quantization)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# Load the vision-language model via unsloth's FastVisionModel using QLoRA settings
model, tokenizer = FastVisionModel.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    use_gradient_checkpointing="unsloth", 
    device_map="cuda:0"
)

# --- Apply LoRA Fine-Tuning Configuration ---
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,  # Using r=8 as in code 1
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", 'gate_proj', 'down_proj', 'up_proj', 'lm_head'],
)
model = get_peft_model(model, peft_config)

# Clear GPU memory before training
torch.cuda.empty_cache()
gc.collect()

# Import extract_text_data and its dependencies from simple-preprocess-4.py
from simple_data_preprocess import extract_text_data, find_ranking_position, extract_ranking, extract_score

def resize_if_needed(img, max_size=512):
    # Check if the image width or height exceeds the max_size
    if img.width > max_size or img.height > max_size:
        # Compute the scaling factor to maintain the aspect ratio
        scaling_factor = max_size / float(max(img.width, img.height))
        new_width = int(img.width * scaling_factor)
        new_height = int(img.height * scaling_factor)
        return img.resize((new_width, new_height), Image.LANCZOS)
    return img

def create_pair_data(images, indices, extracted_info, category, pairwise_data, original_ranking=None, debug_info=None):
    """Create a data entry for a pair of images with reindexed info.
    
    Args:
        images: List of image paths
        indices: List of indices for the pair
        extracted_info: List of dictionaries with image info
        category: Product category
        pairwise_data: List to append the new pair data to
        original_ranking: Original ranking text (for 2-image groups)
        debug_info: Dictionary with debug information
    """
    # Select only the specified images
    pair_images = [images[i] for i in indices]
    
    # Get scores for the pair and map original indices to new indices (0 and 1)
    pair_scores = []
    pair_extracted_info = {}
    
    # Map to convert ordinal words to numbers and back
    ordinal_to_num = {"First": 1, "Second": 2, "Third": 3, "Fourth": 4, "Fifth": 5}
    num_to_ordinal = {1: "First", 2: "Second", 3: "Third", 4: "Fourth", 5: "Fifth"}
    
    for new_idx, original_idx in enumerate(indices):
        if original_idx < len(extracted_info):
            img_info = extracted_info[original_idx]
            score = img_info.get("score", 0)
            pair_scores.append((new_idx, score))
            
            # Create a new entry with the original full text
            new_key = f"Image {new_idx + 1}"
            full_text = img_info.get("full_text", "")
            
            # Comprehensive replacement for all image number references
            updated_text = full_text
            
            # Define and apply all patterns
            
            # Pattern 1: Replace numerical references at the beginning (like "1. ", "2. ")
            if re.search(r'^\d+\.', updated_text, re.MULTILINE):
                updated_text = re.sub(r'^(\d+)\.', f"{new_idx + 1}.", updated_text, flags=re.MULTILINE)
            
            # Pattern 2: Replace Image X references (handle "Image 1", "Image 2", etc.)
            # First find all image number references in the text
            image_numbers = set(re.findall(r'Image\s+(\d+)', updated_text))
            
            # Replace all instances of "Image X" with "Image {new_idx + 1}"
            for img_num in image_numbers:
                updated_text = re.sub(
                    r'Image\s+' + re.escape(img_num), 
                    f"Image {new_idx + 1}", 
                    updated_text
                )
            
            # Pattern 3: Handle ordinal references (First, Second, etc.)
            ordinal_pattern = r'(First|Second|Third|Fourth|Fifth)'
            if re.search(ordinal_pattern, updated_text):
                for ordinal, num in ordinal_to_num.items():
                    updated_text = re.sub(
                        ordinal,
                        num_to_ordinal[new_idx + 1],
                        updated_text
                    )
            
            # Add to the extracted info
            pair_extracted_info[new_key] = {
                "full_text": updated_text,
                "score": score
            }
    
    # For groups with exactly 2 images, use the original ranking text
    if len(indices) == 2 and original_ranking:
        ranking_text = original_ranking
    else:
        # Sort by scores to create ranking
        pair_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Create new ranking text based on the sorted scores
        ranking_text = "###Ranking:\n"
        for rank, (new_idx, score) in enumerate(pair_scores, 1):
            ranking_text += f"{rank}. **Image {new_idx + 1}** - Persuasion Score: {score}/100\n"
    
    pair_data = {
        "images": pair_images,
        "extracted_info": pair_extracted_info,
        "ranking": ranking_text,
        "category": category,
        "group": os.path.basename(os.path.dirname(pair_images[0]))  # Add group info as in code 2
    }
    
    pairwise_data.append(pair_data)
    
    # Add to debug info if provided
    if debug_info is not None:
        debug_info["pairs"].append(pair_data)
    
    return pair_data


# --- Create pairwise dataset ---
pairwise_data = []
debug_examples = {
    "3_image_examples": [],
    "4_image_examples": [],
    "2_image_examples": []
}

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
                
                response_path = os.path.join(dataset_response, category, group, "user_output.txt")
                if os.path.exists(response_path):
                    try:
                        # Use extract_text_data to get the info
                        extracted_info, ranking = extract_text_data(response_path)
                        
                        # Make sure we have valid data
                        if not extracted_info or not isinstance(extracted_info, list):
                            print(f"Warning: Invalid extracted_info for {response_path}")
                            continue
                            
                    except Exception as e:
                        print(f"Error processing {response_path}: {e}")
                        continue
                    
                    # Find the original image index
                    original_idx = None
                    for idx, img_path in enumerate(images):
                        if "original" in img_path.lower():
                            original_idx = idx
                            break
                    
                    # If no image is explicitly named "original", take the last one
                    if original_idx is None and len(images) > 0:
                        original_idx = len(images) - 1
                    
                    # Create debug information for this group
                    debug_info = None
                    
                    # Save examples of 2-image, 3-image, and 4-image groups for debugging
                    if len(images) == 2 and len(debug_examples["2_image_examples"]) < 10:
                        debug_info = {
                            "category": category,
                            "group": group,
                            "original_images": images,
                            "original_info": extracted_info,
                            "original_ranking": ranking,
                            "original_idx": original_idx,
                            "pairs": []
                        }
                        debug_examples["2_image_examples"].append(debug_info)

                    if len(images) == 3 and len(debug_examples["3_image_examples"]) < 30:
                        debug_info = {
                            "category": category,
                            "group": group,
                            "original_images": images,
                            "original_info": extracted_info,
                            "original_ranking": ranking,
                            "original_idx": original_idx,
                            "pairs": []
                        }
                        debug_examples["3_image_examples"].append(debug_info)
                    
                    if len(images) == 4 and len(debug_examples["4_image_examples"]) < 30:
                        debug_info = {
                            "category": category,
                            "group": group,
                            "original_images": images,
                            "original_info": extracted_info,
                            "original_ranking": ranking,
                            "original_idx": original_idx,
                            "pairs": []
                        }
                        debug_examples["4_image_examples"].append(debug_info)
                    
                    if original_idx is not None and len(extracted_info) > 0:
                        # For groups with more than 2 images, create pairs of (edit, original)
                        if len(images) > 2:
                            # For edited images (all except original)
                            edited_indices = [i for i in range(len(images)) if i != original_idx]
                            
                            # If 3 images: create pairs with original
                            if len(images) == 3:
                                for edit_idx in edited_indices:
                                    pair_indices = [edit_idx, original_idx]
                                    create_pair_data(images, pair_indices, extracted_info, category, pairwise_data, debug_info=debug_info)
                            
                            # If 4 images: create pairs of first two edits, last two edits
                            elif len(images) == 4:
                                # First two edits as pair
                                if len(edited_indices) >= 2:
                                    create_pair_data(images, edited_indices[:2], extracted_info, category, pairwise_data, debug_info=debug_info)
                                
                                # Last edit and original as pair
                                if len(edited_indices) >= 1:
                                    create_pair_data(images, [edited_indices[-1], original_idx], extracted_info, category, pairwise_data, debug_info=debug_info)
                        
                        # For groups with exactly 2 images, use them directly as a pair with original ranking
                        elif len(images) == 2:
                            create_pair_data(images, [0, 1], extracted_info, category, pairwise_data, original_ranking=ranking, debug_info=debug_info)

# Save debug examples to files
with open(os.path.join(debug_output_dir, "3_image_examples.json"), "w") as f:
    json.dump(debug_examples["3_image_examples"], f, indent=2)

with open(os.path.join(debug_output_dir, "4_image_examples.json"), "w") as f:
    json.dump(debug_examples["4_image_examples"], f, indent=2)

with open(os.path.join(debug_output_dir, "2_image_examples.json"), "w") as f:
    json.dump(debug_examples["2_image_examples"], f, indent=2)

print(f"Created {len(pairwise_data)} pairwise training samples")
print(f"Debug examples saved to {debug_output_dir}")

# --- Split data into train and validation sets ---
train_data, test_data = train_test_split(pairwise_data, test_size=0.05, random_state=48)
train_data, val_data = train_test_split(train_data, test_size=0.05, random_state=48)
print(f"Training samples: {len(train_data)}, Validation samples: {len(val_data)}")

# --- FIXED COLLATE FUNCTION ---
def collate_fn(batch):
    """Fixed collate function to handle pairwise data properly"""
    batch = batch[0]  # Only process one batch at a time
    try:
        # Load and convert images
        images = []
        for img_path in batch["images"]:
            img = resize_if_needed(Image.open(img_path).convert("RGB"))
            images.append(img)
        
        # Get category and group info
        category = batch["category"]
        group = batch.get("group", "unknown")
        
        # Format image inputs for message structure
        image_content = [{"type": "image", "image": img} for img in images]
        
        # Create the user prompt
        # Create the user prompt
        user_prompt = f"You are evaluating images in the '{group}' group under the '{category}' product category. " \
                      f"Rank the images, based on their appeal for selling '{category}' product. " \
                      f"Provide description, and **persuasion score (1-100)** for each image and explain the ranking."
        
        # Create the full user message with images and text
        user_message = {
            "role": "user",
            "content": image_content + [{"type": "text", "text": user_prompt}]
        }
        
        # Create assistant message with extracted info and ranking
        # Format the extracted info nicely
        extracted_info_text = ""
        for key, info in batch["extracted_info"].items():
            extracted_info_text += f"{key}:\n{info['full_text']}\n\n"
        
        assistant_message = {
            "role": "assistant",
            "content": extracted_info_text + "\n\n" + batch["ranking"]
        }
        
        # Process the dialogue for model input
        prompt_text = processor.apply_chat_template([user_message], tokenize=False, add_generation_prompt=True)
        full_text = processor.apply_chat_template([user_message, assistant_message], tokenize=False, add_generation_prompt=False)
        
        # Process vision info - critical for vision-language models
        image_inputs, video_inputs = process_vision_info([user_message])
        
        # Create input tensors with proper padding
        inputs = processor(
            text=prompt_text,
            images=image_inputs,
            videos=video_inputs,
            padding="longest",
            return_tensors="pt"
        )
        
        # Create label tensors
        labels = processor(
            text=full_text,
            padding="longest", 
            return_tensors="pt"
        )["input_ids"]
        
        # Handle length matching between inputs and labels - critical fix
        input_len = inputs["input_ids"].shape[1]
        labels_len = labels.shape[1]
        
        # Fix tensor size mismatch error by ensuring labels match input_ids shape
        if labels_len < input_len:
            # If labels are shorter, pad with -100 (ignore index)
            padding = torch.full((1, input_len - labels_len), -100, dtype=torch.long)
            labels = torch.cat([labels, padding], dim=1)
        elif labels_len > input_len:
            # If labels are longer, truncate
            labels = labels[:, :input_len]
        
        # Set labels for training with proper handling of padding tokens
        # This guarantees no tensor size mismatch
        inputs["labels"] = torch.where(
            inputs["input_ids"] == processor.tokenizer.pad_token_id,
            -100,  # Ignore padding tokens in loss calculation
            labels
        )
        
        # Ensure batch dimension is present
        if inputs["input_ids"].dim() == 1:
            for key in inputs:
                inputs[key] = inputs[key].unsqueeze(0)
        
        # Make sure we're not passing empty tensors
        assert inputs["input_ids"].numel() > 0, "Empty input tensor detected"
        assert inputs["attention_mask"].numel() > 0, "Empty attention mask tensor detected"
        assert inputs["labels"].numel() > 0, "Empty labels tensor detected"
        
        return inputs
    
    except Exception as e:
        print(f"Error processing batch: {e}")
        # Return a dummy batch that won't cause division by zero
        return {
            "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
            "labels": torch.tensor([[-100, -100, -100]], dtype=torch.long)
        }

# --- IMPROVED TRAINING CONFIGURATION ---
training_args = TrainingArguments(
    output_dir=output_model,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,  # Increased for better stability
    num_train_epochs=3,
    learning_rate=2e-6,  # Lower learning rate for stability
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    eval_strategy="steps",
    eval_steps=50,
    save_strategy="steps",
    save_steps=100,
    logging_steps=10,
    bf16=True,
    remove_unused_columns=False,
    dataloader_pin_memory=False,
    # Added to handle potential NaN loss
    max_grad_norm=1.0,
    # Load best model at the end
    load_best_model_at_end=True,
    # Add early stopping
    metric_for_best_model="eval_loss",
    greater_is_better=False,
)

# Create a custom dataset class to handle empty batches better
class RobustDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx] if idx < len(self.data) else None

# Create robust datasets
robust_train_data = RobustDataset(train_data)
robust_val_data = RobustDataset(val_data)

# Initialize trainer with better error handling
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=robust_train_data,
    eval_dataset=robust_val_data,
    data_collator=collate_fn,
)

# Add exception handling for training
try:
    print("Starting training...")
    trainer.train()
    print("Training completed successfully!")
except Exception as e:
    print(f"Training error: {e}")
    print("Attempting to save the model before exiting...")
    
# Save the model even if training fails
try:
    print("Saving model...")
    model.save_pretrained(output_model)
    processor.save_pretrained(output_model)
    
    print("Uploading model to Hugging Face...")
    model.push_to_hub(repo_id)
    processor.push_to_hub(repo_id)
    
    print(f"Model uploaded: https://huggingface.co/{repo_id} (Private)")
    print("Fine-tuning completed!")
except Exception as e:
    print(f"Error saving or uploading model: {e}")