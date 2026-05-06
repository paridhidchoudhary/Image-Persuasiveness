import os
import torch
import json
from transformers import AutoProcessor, AutoModelForImageTextToText
from huggingface_hub import login
from qwen_vl_utils import process_vision_info
import random 
import torch 
import numpy as np

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
login(token='hf_xxxxxxxxxxxxxxxxx')

# Define model name
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
# Automatically map model to available devices
# model = AutoModelForImageTextToText.from_pretrained(
#     model_name, device_map="auto" # Enable FP16 to reduce memory usage
# )
# model = AutoModelForImageTextToText.from_pretrained(model_name).to(device)
print("Model and processor loaded successfully!")

## SET PATHS
temp_folder = "home/debajyoti/paridhi_mtp/fewshot_for_deep/images"
#temp_folder = "/home/deepg/NAS/Downloads/MTP-2-persuasion/fewshot_for_deep/images"  # For one-shot examples
#score_folder = "/home/deepg/NAS/Downloads/MTP-2-persuasion/fewshot_for_deep/scores"  # Corresponding scores for one-shot 
score_folder = "home/debajyoti/paridhi_mtp/fewshot_for_deep/scores"
# input_root = "/home/deepg/NAS/Downloads/MTP-2-persuasion/dataset/dataset_image copy"  # Evaluation images
# output_root = "/home/deepg/NAS/Downloads/MTP-2-persuasion/dataset/dataset_response copy"  # Where results are saved
input_root = "home/debajyoti/paridhi_mtp/product_images_real/dataset_image_new"
output_root = "home/debajyoti/paridhi_mtp/product_images_real/dataset_response_new"


first_case_saved = False  # Flag to save only the first prompt separately

def get_one_shot_example(category):
    """Fetches the first group in `temp/` and its corresponding score from `scores/`."""
    temp_category_path = os.path.join(temp_folder, category)
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

    # ✅ Fix: First sort filenames, then create dictionaries
    image_files = sorted([
        img for img in os.listdir(group_path)
        if img.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
    ])
    
    images = [{"type": "image", "image": f"file://{os.path.join(group_path, img)}"} for img in image_files]

    # Read the corresponding `score.txt`
    with open(score_path, "r", encoding="utf-8") as f:
        score_text = f.read().strip()

    print(f"One-shot example from '{first_group}' in '{category}' with {len(images)} images.")

    return {"group": first_group, "images": images, "score": score_text}

def get_evaluation_groups(category):
    """Fetch all new groups from `input_root/` for evaluation."""
    input_category_path = os.path.join(input_root, category)
    evaluation_groups = []

    if not os.path.isdir(input_category_path):
        return []

    for group in sorted(os.listdir(input_category_path)):
        group_path = os.path.join(input_category_path, group)
        if os.path.isdir(group_path):
            evaluation_groups.append(group_path)

    return evaluation_groups

def get_free_gpu_memory():
    """Returns the available GPU memory in GB."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        return torch.cuda.mem_get_info()[0] / (1024 ** 3)  # Convert bytes to GB
    return 0  # If no GPU, return 0

def process_images():
    """Processes images and saves results in structured category/group folders."""
    global first_case_saved
    
    #for category in os.listdir(temp_folder):
    for category in [c for c in os.listdir(temp_folder) if c in ["chair", "couch"]]:
        category_path = os.path.join(temp_folder, category)
        if os.path.isdir(category_path):

            # ✅ Get the one-shot example (first group inside category folder in `temp/`)
            one_shot_example = get_one_shot_example(category)
            if not one_shot_example:
                print(f"No valid example found for category '{category}', skipping...")
                continue

            # ✅ Fetch new groups inside this category from `input_root/`
            evaluation_groups = get_evaluation_groups(category)

            # Process each evaluation group
            for group_path in evaluation_groups:
                group = os.path.basename(group_path)

                # ✅ Fix: First sort filenames, then create dictionaries
                image_files = sorted([
                    img for img in os.listdir(group_path)
                    if img.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
                ])

                new_images = [{"type": "image", "image": f"file://{os.path.join(group_path, img)}"} for img in image_files]

                # ✅ Check total number of images (Few-shot + New)
                total_images = len(one_shot_example["images"]) + len(new_images)


                print(f"Processing {len(new_images)} images in group '{group}' ({category})...")

                torch.cuda.empty_cache()  # Free memory before running inference

                prompt_items = []
                # ✅ Add one-shot example (Images + Score)
                print(f"Using one-shot example from '{one_shot_example['group']}' in '{category}'")
                prompt_items.append({
                    "type": "text",
                    "text": f"Example from group '{one_shot_example['group']}' in '{category}' category:"
                })
                prompt_items.extend(one_shot_example["images"])
                prompt_items.append({
                    "type": "text",
                    "text": f"Example persuasion score and explanation for group {one_shot_example['group']}:\n{one_shot_example['score']}"
                })
                
                # ✅ Add new images for evaluation
                evaluation_prompt = (
                    f"\nNow, evaluate new images in the '{group}' group under the '{category}' category. "
                    f"Rank the images **within this group only**, based on their appeal for selling this specific type of product. "
                    f"Provide a description, and **persuasion score (1-100)** on each image, along with an explanation for your ranking."
                )
                prompt_items.append({"type": "text", "text": evaluation_prompt})
                prompt_items.extend(new_images)
                messages = [{"role": "user", "content": prompt_items}]
                
                prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = processor(
                    text=[prompt_text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                ).to(device)

                # ✅ Handle CUDA Out of Memory error safely
                try:
                    with torch.no_grad():
                        generated_ids = model.generate(**inputs, max_new_tokens=1200)
                except torch.cuda.OutOfMemoryError:
                    print(f"Skipping group '{group}' ({category}) due to CUDA OOM error. Freeing memory and continuing...")
                    torch.cuda.empty_cache()
                    continue

                output_text = processor.batch_decode(
                    generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0]

                # ✅ Remove prompt, keep only model response
                cleaned_output_text = output_text.split("assistant\n", 1)[-1].strip()

                # ✅ Save response
                os.makedirs(os.path.join(output_root, category, group), exist_ok=True)
                with open(os.path.join(output_root, category, group, "output_qwen_fewshot.txt"), "w", encoding="utf-8") as f:
                    f.write(cleaned_output_text)

                print(f"Response saved for {group} ({category})")


if __name__ == "__main__":
    process_images()
