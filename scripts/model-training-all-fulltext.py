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
import random 
import torch 
import numpy as np
SEED = 50
random.seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

import torch._dynamo
torch._dynamo.config.suppress_errors = True
# OR completely disable dynamo
import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"

# --- Authenticate with Hugging Face ---
login(token="hf_xxxxxxxxxxxxxxxxxxxx")

## SET PATHS
#data_root = "/home/deepg/NAS/Downloads/MTP-2-persuasion/dataset"
data_root = "home/debajyoti/paridhi_mtp/product_images_real"
dataset_image = os.path.join(data_root, "dataset_image_new")
dataset_response = os.path.join(data_root, "final_data")
output_model = "./vlm_finetuned_full_context_48_2"
MAX_IMAGES = 4  # Matching the evaluation code max images


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
    r=8,  # Increased from 8 to 16 for better capacity
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", 'gate_proj', 'down_proj', 'up_proj'],
    #target_modules=["q_proj", "v_proj", "k_proj", "o_proj", 'gate_proj', 'down_proj', 'up_proj','lm_head'],
)
model = get_peft_model(model, peft_config)

# Clear GPU memory before training
torch.cuda.empty_cache()
gc.collect()

# --- Load Dataset ---
data = []

# Import extract_text_data from simple-preprocess-4.py
from simple_data_preprocess import extract_text_data

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
                
                if len(images) > MAX_IMAGES:
                    continue
                    
                # response_path = os.path.join(dataset_response, category, group, "output_pixtral_zeroshot.txt")
                response_path = os.path.join(dataset_response, category, group, "user_output.txt")
                if os.path.exists(response_path):
                    # Read the full response text instead of extracting
                    with open(response_path, 'r') as f:
                        full_response_text = f.read()
                    
                    # Still extract ranking for dataset info, but we won't use extracted_info
                    extracted_info, ranking = extract_text_data(response_path)
                    
                    data.append({
                        "images": images,
                        "full_response_text": full_response_text,  # Store full text
                        "ranking": ranking,
                        "category": category,
                        "group": group
                    })

print(f"Dataset loaded with {len(data)} groups")
train_data, test_data = train_test_split(data, test_size=0.05, random_state=48)
train_data, val_data = train_test_split(train_data, test_size=0.05, random_state=48)
print(f"Training dataset: {len(train_data)} groups, Validation dataset: {len(val_data)} groups")

def resize_if_needed(img, max_size=384):  # Increased max size for better image quality
    # Check if the image width or height exceeds the max_size
    if img.width > max_size or img.height > max_size:
        # Compute the scaling factor to maintain the aspect ratio
        scaling_factor = max_size / float(max(img.width, img.height))
        new_width = int(img.width * scaling_factor)
        new_height = int(img.height * scaling_factor)
        return img.resize((new_width, new_height), Image.LANCZOS)
    return img

# def collate_fn(batch):
#     batch = batch[0]
#     try:
#         # Format image paths for processing
#         sorted_images = []
#         for img_path in batch["images"]:
#             # Load and resize image
#             img = resize_if_needed(Image.open(img_path).convert("RGB"))
#             sorted_images.append({"type": "image", "image": img})
        
#         # Create detailed prompt
#         category = batch["category"]
#         group = batch["group"]
        
#         messages = [
#             {
#                 "role": "user",
#                 "content": sorted_images + [  # Images first, then text
#                     {
#                         "type": "text", 
#                         "text": f"You are evaluating images in the '{group}' group under the '{category}' product category. "
#                                 f"Rank the images, based on their appeal for selling '{category}' product. "
#                                 f"Provide description, and *persuasion score (1-100)* for each image and explain the ranking."
#                     }
#                 ],
#             }
#         ]
        
#         # Use the full response text for the assistant message
#         full_response_text = batch["full_response_text"]
        
#         assistant_message = {
#             "role": "assistant",
#             "content": full_response_text
#         }
        
#         # Prepare inputs
#         prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
#         full_text = processor.apply_chat_template(messages + [assistant_message], tokenize=False, add_generation_prompt=False)
        
#         # Process vision info
#         image_inputs, video_inputs = process_vision_info(messages)
        
#         # Create input tensors
#         inputs = processor(
#             text=prompt_text,
#             images=image_inputs,
#             videos=video_inputs,
#             padding=True,
#             return_tensors="pt",
#         )
        
#         # Create label tensors
#         labels = processor(
#             text=full_text,
#             padding=True,
#             return_tensors="pt"
#         )["input_ids"]
        
#         # Handle potential length mismatches
#         input_len = inputs["input_ids"].shape[1]
#         labels_len = labels.shape[1]
        
#         if labels_len < input_len:
#             padding = torch.full((1, input_len - labels_len), -100, dtype=torch.long)
#             labels = torch.cat([labels, padding], dim=-1)
#         elif labels_len > input_len:
#             labels = labels[:, :input_len]
            
#         # Set labels for training
#         inputs["labels"] = torch.where(
#             inputs["input_ids"] == processor.tokenizer.pad_token_id,
#             -100,
#             labels
#         )
#        # inputs["num_items_in_batch"] = len(batch["images"]) if "images" in batch else 1

        
#         return inputs
#     except Exception as e:
#         print(f"Error in collate function: {e}")
#         torch.cuda.empty_cache()
#         gc.collect()
#         return {"input_ids": torch.tensor([[0]]), "attention_mask": torch.tensor([[0]]), "labels": torch.tensor([[-100]])}

def collate_fn(batch):
    batch = batch[0]
    try:
        # Format image paths for processing
        sorted_images = []
        for img_path in batch["images"]:
            # Load and resize image
            img = resize_if_needed(Image.open(img_path).convert("RGB"))
            sorted_images.append({"type": "image", "image": img})
        
        # Create detailed prompt
        category = batch["category"]
        group = batch["group"]
        
        messages = [
            {
                "role": "user",
                "content": sorted_images + [  # Images first, then text
                    {
                        "type": "text", 
                        "text": f"You are evaluating images in the '{group}' group under the '{category}' product category. "
                                f"Rank the images, based on their appeal for selling '{category}' product. "
                                f"Provide description, and *persuasion score (1-100)* for each image and explain the ranking."
                    }
                ],
            }
        ]
        
        # Use the full response text for the assistant message
        full_response_text = batch["full_response_text"]
        
        assistant_message = {
            "role": "assistant",
            "content": full_response_text
        }
        
        # Prepare inputs
        prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        full_text = processor.apply_chat_template(messages + [assistant_message], tokenize=False, add_generation_prompt=False)
        
        # Process vision info
        image_inputs, video_inputs = process_vision_info(messages)
        
        # Create input tensors
        inputs = processor(
            text=prompt_text,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        
        # Create label tensors
        labels = processor(
            text=full_text,
            padding=True,
            return_tensors="pt"
        )["input_ids"]
        
        # Handle potential length mismatches
        input_len = inputs["input_ids"].shape[1]
        labels_len = labels.shape[1]
        
        if labels_len < input_len:
            padding = torch.full((1, input_len - labels_len), -100, dtype=torch.long)
            labels = torch.cat([labels, padding], dim=-1)
        elif labels_len > input_len:
            labels = labels[:, :input_len]
            
        # Set labels for training
        inputs["labels"] = torch.where(
            inputs["input_ids"] == processor.tokenizer.pad_token_id,
            -100,
            labels
        )
        
        # CRITICAL FIX: Add num_items_in_batch as an integer, not inside inputs dict
        # This should be returned at the top level of the batch
        inputs["num_items_in_batch"] = torch.tensor([len(batch["images"])], dtype=torch.long)
        
        return inputs
    except Exception as e:
        print(f"Error in collate function: {e}")
        torch.cuda.empty_cache()
        gc.collect()
        return {
            "input_ids": torch.tensor([[0]]), 
            "attention_mask": torch.tensor([[0]]), 
            "labels": torch.tensor([[-100]]),
            "num_items_in_batch": torch.tensor([1], dtype=torch.long)  # Add this to error case too
        }

# --- Training Configuration ---
training_args = TrainingArguments(
    output_dir=output_model,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,  # Increased for better stability
    num_train_epochs=3,  # Increased for better learning
    learning_rate=2e-6,  # Slightly lower learning rate for stability
    lr_scheduler_type="cosine",  # Added cosine scheduler
    warmup_ratio=0.1,  # Added warmup
    eval_strategy="steps",
    eval_steps=50,
    save_strategy="steps",
    save_steps=100,
    logging_steps=10,
    bf16=True,
    remove_unused_columns=False,
    dataloader_pin_memory=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=val_data,
    data_collator=collate_fn,
)

# --- Fine-tune the Model ---
print("Starting training...")
trainer.train()

# --- Save and Upload the Fine-tuned Model ---
print("Saving model...")
model.save_pretrained(output_model)
processor.save_pretrained(output_model)