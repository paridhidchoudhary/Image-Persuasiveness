import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import unsloth
from unsloth import FastVisionModel
import json
import torch
import gc
import re
from PIL import Image
from sklearn.model_selection import train_test_split
from transformers import AutoProcessor, TrainingArguments, Trainer, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType
from huggingface_hub import login, HfApi
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
import argparse
import random 
import numpy as np
import torch.nn as nn

torch._dynamo.config.suppress_errors = True

# Set seed for reproducibility
SEED = 50
random.seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

def resize_if_needed(img, max_size=256):
    """Resize image if it exceeds max dimensions while preserving aspect ratio"""
    if img.width > max_size or img.height > max_size:
        scaling_factor = max_size / float(max(img.width, img.height))
        new_width = int(img.width * scaling_factor)
        new_height = int(img.height * scaling_factor)
        return img.resize((new_width, new_height), Image.LANCZOS)
    return img

def collate_fn(batch):
    """Collate function for dataloader that incorporates both images and M1 outputs"""
    batch = batch[0]
    try:
        # Format image paths for processing
        sorted_images = []
        for img_path in batch["images"]:
            # Load and resize image
            img = resize_if_needed(Image.open(img_path).convert("RGB"))
            sorted_images.append({"type": "image", "image": img})
        
        # Create detailed prompt including M1 outputs
        category = batch["category"]
        group = batch["group"]
        m1_output = batch["m1_output"]
        
        messages = [
            {
                "role": "user",
                "content": sorted_images + [  # Images first, then text
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
        
        # Use the ground truth for the final assistant message
        ground_truth_text = batch["ground_truth"]
        
        assistant_message = {
            "role": "assistant",
            "content": ground_truth_text
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
            truncation=True,
            max_length=2048,
            return_tensors="pt",
        )
        
        # Create label tensors
        labels = processor(
            text=full_text,
            padding=True,
            truncation=True,
            max_length=2048,
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
        
        # CRITICAL FIX: Add num_items_in_batch
        inputs["num_items_in_batch"] = torch.tensor([len(batch["images"])], dtype=torch.long)
        
        return inputs
    except Exception as e:
        print(f"Error in collate function: {e}")
        import traceback
        traceback.print_exc()
        torch.cuda.empty_cache()
        gc.collect()
        return {
            "input_ids": torch.tensor([[0]]), 
            "attention_mask": torch.tensor([[0]]), 
            "labels": torch.tensor([[-100]]),
            "num_items_in_batch": torch.tensor([1], dtype=torch.long)
        }

# Custom Trainer to handle loss computation
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Override to use standard PyTorch cross-entropy"""
        inputs.pop("num_items_in_batch", None)
        
        labels = inputs.pop("labels")
        
        outputs = model(**inputs)
        logits = outputs.logits
        
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        shift_labels = shift_labels.view(-1)
        
        loss = loss_fct(shift_logits, shift_labels)
        
        return (loss, outputs) if return_outputs else loss

def main():
    
    # Authenticate with Hugging Face if token provided
    login(token="hf_xxxxxxxxxxxxx")

    ## SET PATHS
    data_root = "home/debajyoti/paridhi_mtp/product_images_real"
    dataset_image = os.path.join(data_root, "dataset_image_new")

    base_model = "./vlm_finetuned_full_context-Merged-Model_2"
    output_model = "./vlm_finetuned_full_context_47_1_m2"
    dataset_response = os.path.join(data_root, "final_data")
    m1_outputs_dir = os.path.join(data_root, "m1_inference_outputs")
    
    # Create output directory
    os.makedirs(output_model, exist_ok=True)
    
    # Load dataset from saved M1 outputs
    print("Loading dataset with M1 outputs...")
    data = []
    
    # Traverse M1 outputs directory structure
    for category in os.listdir(m1_outputs_dir):
        category_path = os.path.join(m1_outputs_dir, category)
        if os.path.isdir(category_path):
            for group in os.listdir(category_path):
                group_output_path = os.path.join(m1_outputs_dir, category, group)
                if os.path.isdir(group_output_path):
                    # Check if M1 output and ground truth exist
                    m1_output_path = os.path.join(group_output_path, "m1_output.txt")
                    ground_truth_path = os.path.join(group_output_path, "ground_truth.txt")
                    
                    if os.path.exists(m1_output_path) and os.path.exists(ground_truth_path):
                        # Get original image paths
                        original_image_dir = os.path.join(dataset_image, category, group)
                        if os.path.isdir(original_image_dir):
                            images = sorted([
                                os.path.join(original_image_dir, img)
                                for img in os.listdir(original_image_dir)
                                if img.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
                            ])
                            
                            if images:
                                # Read M1 output and ground truth
                                with open(m1_output_path, 'r') as f:
                                    m1_output = f.read()
                                
                                with open(ground_truth_path, 'r') as f:
                                    ground_truth = f.read()
                                
                                # Truncate if too long
                                if len(m1_output) > 1500:
                                    m1_output = m1_output[:1500]
                                if len(ground_truth) > 2000:
                                    ground_truth = ground_truth[:2000]
                                
                                data.append({
                                    "images": images,
                                    "m1_output": m1_output,
                                    "ground_truth": ground_truth,
                                    "category": category,
                                    "group": group
                                })
    
    print(f"Dataset loaded with {len(data)} groups")
    
    # Split data into training and validation sets
    train_data, test_data = train_test_split(
        data, 
        test_size=0.05, 
        random_state=47
    )
    train_data, val_data = train_test_split(
        train_data, 
        test_size=0.05, 
        random_state=47
    )
    print(f"Training dataset: {len(train_data)} groups, Validation dataset: {len(val_data)} groups")
    
    # --- Load Base Model ---
    print(f"Loading base model: {base_model}")
    
    # Global variables needed by collate_fn
    global processor
    processor = AutoProcessor.from_pretrained(base_model, use_fast=False)
    
    # Create BitsAndBytesConfig for QLoRA (4-bit quantization)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    torch.cuda.empty_cache()
    gc.collect()
    
    # Load the vision-language model for M2
    model, tokenizer = FastVisionModel.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        use_gradient_checkpointing="unsloth", 
        device_map="auto",
        max_seq_length=2048,
    )
    
    if hasattr(model.config, 'use_fused_cross_entropy'):
        model.config.use_fused_cross_entropy = False
    
    model.config.use_cache = False
    
    # --- Apply LoRA Fine-Tuning Configuration ---
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=4,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
    )
    model = get_peft_model(model, peft_config)
    
    # Clear GPU memory before training
    torch.cuda.empty_cache()
    gc.collect()
    
    # --- Training Configuration ---
    training_args = TrainingArguments(
        output_dir=output_model,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=2,
        learning_rate=2e-6,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        logging_steps=10,
        fp16=True,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        torch_compile=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
        max_grad_norm=0.3,
        dataloader_num_workers=0,
    )
    
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        data_collator=collate_fn,
    )
    
    # --- Fine-tune the Model ---
    print("Starting M2 training...")
    print(f"GPU Memory: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
    
    try:
        trainer.train()
        print("Training completed successfully!")
    except Exception as e:
        print(f"Training error: {e}")
        import traceback
        traceback.print_exc()
    
    # --- Save the Fine-tuned Model ---
    print(f"Saving M2 model to {output_model}...")
    model.save_pretrained(output_model)
    processor.save_pretrained(output_model)
    
    print("M2 fine-tuning completed!")

if __name__ == "__main__":
    main()