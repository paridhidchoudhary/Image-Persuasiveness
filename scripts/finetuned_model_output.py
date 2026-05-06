import unsloth
from unsloth import FastVisionModel
import os
import torch
import gc
import re
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, BitsAndBytesConfig
from huggingface_hub import login
from qwen_vl_utils import process_vision_info  # Required for vision processing
import argparse
import random 
import numpy as np

# Set seed for reproducibility
SEED = 50
random.seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

def resize_if_needed(img, max_size=384):
    """Resize image if it exceeds max dimensions while preserving aspect ratio"""
    if img.width > max_size or img.height > max_size:
        scaling_factor = max_size / float(max(img.width, img.height))
        new_width = int(img.width * scaling_factor)
        new_height = int(img.height * scaling_factor)
        return img.resize((new_width, new_height), Image.LANCZOS)
    return img

def generate_m1_output(model, processor, category, group, image_paths):
    """Generate explanations and rankings using M1 model for a group of images"""
    try:
        # Prepare images for inference
        sorted_images = []
        for img_path in image_paths:
            img = resize_if_needed(Image.open(img_path).convert("RGB"))
            sorted_images.append({"type": "image", "image": img})
        
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
            }
        ]
        
        # Process vision info
        image_inputs, video_inputs = process_vision_info(messages)
        
        # Create input for model
        inputs = processor(
            text=processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True),
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(model.device)
        
        # Generate output from M1
        with torch.no_grad():
            output_tokens = model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=False,
            )
        
        # Decode output
        m1_output = processor.batch_decode(output_tokens, skip_special_tokens=True)[0]
        
        # Extract and return only the assistant's response
        assistant_response = m1_output.split("ASSISTANT: ")[-1].strip()
        return assistant_response
        
    except Exception as e:
        print(f"Error generating M1 output for {category}/{group}: {e}")
        return f"Error: {e}"

def main():
    
    # Authenticate with Hugging Face if token provided
    login(token="hf_xxxxxxxxxxxx")

    ## SET PATHS
    data_root = "home/debajyoti/paridhi_mtp/product_images_real"
    dataset_image = os.path.join(data_root, "dataset_image_new")
    dataset_response = os.path.join(data_root, "final_data")

    m1_outputs_dir = os.path.join(data_root, "m1_inference_outputs")
    
    # Create output directory
    os.makedirs(m1_outputs_dir, exist_ok=True)
    m1_model_path = "./vlm_finetuned_full_context_48_2"
    # Load M1 model
    print("Loading M1 model...")
    processor = AutoProcessor.from_pretrained(m1_model_path, use_fast=False)
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    model, _ = FastVisionModel.from_pretrained(
        m1_model_path,
        quantization_config=bnb_config,
        device_map="auto"
    )
    model.eval()
    
    # Find all valid groups with ground truth
    groups_to_process = []
    
    for category in os.listdir(dataset_image):
        category_path = os.path.join(dataset_image, category)
        if os.path.isdir(category_path):
            for group in os.listdir(category_path):
                # Check if there's a ground truth response for this group
                user_output_path = os.path.join(dataset_response, category, group, "user_output.txt")
                group_path = os.path.join(dataset_image, category, group)
                
                if os.path.isdir(group_path) and os.path.exists(user_output_path):
                    # Get image paths
                    images = sorted([
                        os.path.join(group_path, img)
                        for img in os.listdir(group_path)
                        if img.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
                    ])
                    
                    if images and len(images) <= 4:
                        groups_to_process.append((category, group, images, user_output_path))
    
    
    print(f"Found {len(groups_to_process)} valid groups to process")
    
    # Process each group
    for idx, (category, group, images, user_output_path) in enumerate(tqdm(groups_to_process, desc="Processing groups")):
        # Create output directory structure if it doesn't exist
        output_dir = os.path.join(m1_outputs_dir, category, group)
        os.makedirs(output_dir, exist_ok=True)
        
        # Check if M1 output already exists
        m1_output_path = os.path.join(output_dir, "m1_output.txt")
        
        
        # Generate M1 output
        print(f"Processing {idx+1}/{len(groups_to_process)}: {category}/{group}")
        m1_output = generate_m1_output(model, processor, category, group, images)
        
        # Save M1 output
        with open(m1_output_path, 'w') as f:
            f.write(m1_output)
        
        # Copy ground truth to the same directory for convenience
        with open(user_output_path, 'r') as f:
            ground_truth = f.read()
        
        with open(os.path.join(output_dir, "ground_truth.txt"), 'w') as f:
            f.write(ground_truth)
    
    print(f"All M1 outputs generated and saved to {m1_outputs_dir}")

if __name__ == "__main__":
    main()