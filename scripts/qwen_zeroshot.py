import os
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from huggingface_hub import login
from qwen_vl_utils import process_vision_info
import random 
import torch 
import numpy as np
import json

#memory optimizations - can remove if enough gpu memory
# Also add this at the very beginning after imports:
import gc
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# Set memory allocation strategy
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512,expandable_segments:True'

#original
SEED = 50
random.seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

# checkpointing - added by paridhi
# ADD THESE CHECKPOINT FUNCTIONS HERE
def is_already_processed(output_path):
    return os.path.exists(output_path)

def save_progress(category, group, progress_file="progress.json"):
    progress = {"last_category": category, "last_group": group}
    with open(progress_file, "w") as f:
        json.dump(progress, f)
    print(f"Progress saved: {category}/{group}")

def load_progress(progress_file="progress.json"):
    if os.path.exists(progress_file):
        with open(progress_file, "r") as f:
            return json.load(f)
    return None


# Authenticate with Hugging Face
login(token='hf_xxxxxxxxxxx')

# # Define model name
model_name = "qwen/Qwen2.5-VL-7B-Instruct"

#memory optimization
# Replace the model loading section with this:
model = AutoModelForImageTextToText.from_pretrained(
    model_name, 
    torch_dtype=torch.float16,  # Use half precision
    low_cpu_mem_usage=True,     # Load efficiently
    device_map="auto"           # Automatically distribute across available GPUs/CPU
)
# Load model and processor
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = AutoProcessor.from_pretrained(model_name, revision="refs/pr/24", use_fast=False)
#model = AutoModelForImageTextToText.from_pretrained(model_name).to(device)

print("Model and processor loaded successfully!")

## SET PATHS
input_root_folder = "home/debajyoti/paridhi_mtp/product_images_real/dataset_image_new"
#input_root_folder = "/home/deepg/NAS/Downloads/MTP-2-persuasion/dataset/dataset_image copy"
#output_root_folder = "/home/deepg/NAS/Downloads/MTP-2-persuasion/dataset/dataset_response copy"
output_root_folder = "home/debajyoti/paridhi_mtp/product_images_real/dataset_response_new"

# MODIFY YOUR EXISTING LOOP - REPLACE YOUR EXISTING FOR LOOPS WITH THIS:
progress = load_progress() or {"last_category": None, "last_group": None}
start_processing = progress["last_category"] is None

#for category in os.listdir(input_root_folder):
for category in [c for c in os.listdir(input_root_folder) if c in ["remote"]]:
    # Skip categories until we reach the last processed one
    if not start_processing and category != progress["last_category"]:
        print(f"Skipping category: {category}")
        continue
        
    category_path = os.path.join(input_root_folder, category)
    if os.path.isdir(category_path):
        print(f"Processing category: {category}")

        # Iterate over groups within each category
        for group in os.listdir(category_path):
            # Skip groups until we reach the last processed one
            if not start_processing and group != progress["last_group"]:
                print(f"Skipping group: {group}")
                continue
            start_processing = True  # Start processing from this point onwards
            
            group_path = os.path.join(category_path, group)
            if os.path.isdir(group_path):
                
                # CHECK IF ALREADY PROCESSED - ADD THIS
                txt_filename = os.path.join(output_root_folder, category, group, "output_qwen_zeroshot.txt")
                if is_already_processed(txt_filename):
                    print(f"✓ Skipping {category}/{group} - already processed")
                    continue

                images = []
                image_files = sorted(os.listdir(group_path))  # ✅ Ensuring sorted order

                # Collect all images in the group
                for image_file in image_files:
                    image_path = os.path.join(group_path, image_file)
                    images.append({"type": "image", "image": f"file://{image_path}"})
 
                print(f"Processing group: {group}")

                # Create a multi-image inference prompt with category & group information
                messages = [
                    {
                        "role": "user",
                        "content": images + [
                            {
                                "type": "text",
                                "text": f"You are evaluating images in the '{group}' group under the '{category}' product category. "
                                        f"Rank the images, based on their appeal for selling this specific type of product. "
                                        f"Provide a description, and **persuasion score (1-100)** for each image and explain the ranking."
                            }
                        ],
                    }
                ]

                # Free unused GPU memory
                torch.cuda.empty_cache()

                # Prepare inputs
                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                ).to(device)

                # Inference
                try:
                    with torch.no_grad():
                        generated_ids = model.generate(**inputs, max_new_tokens=1200)
                except torch.cuda.OutOfMemoryError:
                    print(f"Skipping group '{group}' ({category}) due to CUDA OOM error. Freeing memory and continuing...")
                    torch.cuda.empty_cache()
                    continue
                    
                # Extract raw model output
                raw_output_text = processor.batch_decode(
                    generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0]

                # Remove everything before "assistant\n"
                cleaned_output_text = raw_output_text.split("assistant\n", 1)[-1].strip()

                # Define output folder structure matching input
                category_folder = os.path.join(output_root_folder, category)
                group_folder = os.path.join(category_folder, group)

                # Create folders if they don't exist
                os.makedirs(group_folder, exist_ok=True)

                # Define TXT output file path
                txt_filename = os.path.join(group_folder, "output_qwen_zeroshot.txt")

                # Save only the cleaned assistant response as TXT
                with open(txt_filename, "w", encoding="utf-8") as txt_file:
                    txt_file.write(cleaned_output_text)

                print(f"✓ Response saved: {txt_filename}")
                
                # SAVE PROGRESS AFTER EACH GROUP - ADD THIS
                save_progress(category, group)
                
                # BETTER MEMORY CLEANUP - ADD this
                del generated_ids, inputs
                torch.cuda.empty_cache()
                import gc
                gc.collect()

print("All processing completed!")
# CLEAN UP PROGRESS FILE WHEN DONE - ADD THIS
if os.path.exists("progress.json"):
    os.remove("progress.json")
    print("Progress file cleaned up.")


# # Original code for reference
# # Iterate over product categories (TV, Laptop, etc.)
# for category in os.listdir(input_root_folder):
#     category_path = os.path.join(input_root_folder, category)
#     if os.path.isdir(category_path):

#         # Iterate over groups within each category
#         for group in os.listdir(category_path):
#             group_path = os.path.join(category_path, group)
#             if os.path.isdir(group_path):

#                 images = []
#                 image_files = sorted(os.listdir(group_path))  # ✅ Ensuring sorted order

#                 # Collect all images in the group
#                 for image_file in image_files:
#                     image_path = os.path.join(group_path, image_file)
#                     images.append({"type": "image", "image": f"file://{image_path}"})
 

#                 print(f"Processing group: {group}")

#                 # Create a multi-image inference prompt with category & group information
#                 messages = [
#                     {
#                         "role": "user",
#                         "content": images + [
#                             {
#                                 "type": "text",
#                                 "text": f"You are evaluating images in the '{group}' group under the '{category}' product category. "
#                                         f"Rank the images, based on their appeal for selling this specific type of product. "
#                                         f"Provide a description, and **persuasion score (1-100)** for each image and explain the ranking."
#                             }
#                         ],
#                     }
#                 ]

#                 # Free unused GPU memory
#                 torch.cuda.empty_cache()

#                 # Prepare inputs
#                 text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
#                 image_inputs, video_inputs = process_vision_info(messages)
#                 inputs = processor(
#                     text=[text],
#                     images=image_inputs,
#                     videos=video_inputs,
#                     padding=True,
#                     return_tensors="pt",
#                 ).to(device)

#                 # Inference

#                 try:
#                     with torch.no_grad():
#                         generated_ids = model.generate(**inputs, max_new_tokens=1200)
#                 except torch.cuda.OutOfMemoryError:
#                     print(f"Skipping group '{group}' ({category}) due to CUDA OOM error. Freeing memory and continuing...")
#                     torch.cuda.empty_cache()
#                     continue
#                 # Extract raw model output
#                 raw_output_text = processor.batch_decode(
#                     generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
#                 )[0]

#                 # Remove everything before "assistant\n"
#                 cleaned_output_text = raw_output_text.split("assistant\n", 1)[-1].strip()

#                 # Define output folder structure matching input
#                 category_folder = os.path.join(output_root_folder, category)
#                 group_folder = os.path.join(category_folder, group)

#                 # Create folders if they don’t exist
#                 os.makedirs(group_folder, exist_ok=True)

#                 # Define TXT output file path
#                 txt_filename = os.path.join(group_folder, "output_qwen_zeroshot.txt")

#                 # Save only the cleaned assistant response as TXT
#                 with open(txt_filename, "w", encoding="utf-8") as txt_file:
#                     txt_file.write(cleaned_output_text)

#                 print(f"Cleaned response saved in {txt_filename}")
