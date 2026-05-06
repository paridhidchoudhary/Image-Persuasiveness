import os
import base64
import json
from mistralai import Mistral
import random 
import torch 
import numpy as np
SEED = 50
random.seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
# === Configuration ===
API_KEY = "MgFASHu9FVkLBk2QRPqTL70trCY0QqoZ"
MODEL_NAME = "pixtral-large-latest"

## SET PATHS
ROOT_FOLDER = "home/debajyoti/paridhi_mtp/product_images_real/dataset_image_new"
#ROOT_FOLDER = "/home/deepg/NAS/Downloads/MTP-2-persuasion/dataset/dataset_image copy"
# Folder with few-shot examples.
FEWSHOT_FOLDER = "home/debajyoti/paridhi_mtp/fewshot_for_deep/images"
#FEWSHOT_FOLDER = "/home/deepg/NAS/Downloads/MTP-2-persuasion/fewshot_for_deep/images"
# Folder with score text files corresponding to the few-shot examples.
SCORE_FOLDER = "home/debajyoti/paridhi_mtp/fewshot_for_deep/scores"
#SCORE_FOLDER = "/home/deepg/NAS/Downloads/MTP-2-persuasion/fewshot_for_deep/scores"
# Folder to store results.
OUTPUT_FOLDER = "home/debajyoti/paridhi_mtp/product_images_real/dataset_response_new"
#OUTPUT_FOLDER = "/home/deepg/NAS/Downloads/MTP-2-persuasion/dataset/dataset_response copy"

# === Initialize Mistral client ===
client = Mistral(api_key=API_KEY)

# === Utility Functions ===

def encode_image(image_path):
    """Encodes an image file as a base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def get_images(folder_path):
    """Returns a sorted list of image file paths in a folder."""
    return sorted([
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
    ])

def load_fewshot_examples(category):
    """
    Loads the first two groups in FEWSHOT_FOLDER/category as few-shot examples.
    Retrieves all images in those groups and their corresponding scores from SCORE_FOLDER.
    """
    fewshot_category_path = os.path.join(FEWSHOT_FOLDER, category)
    score_category_path = os.path.join(SCORE_FOLDER, category)

    if not os.path.isdir(fewshot_category_path) or not os.path.isdir(score_category_path):
        return None

    groups = sorted(os.listdir(fewshot_category_path))[:2]  # Select first two groups
    if not groups:
        return None

    fewshot_examples = []

    for group in groups:
        group_path = os.path.join(fewshot_category_path, group)
        score_path = os.path.join(score_category_path, group, "score.txt")

        if not os.path.isdir(group_path) or not os.path.exists(score_path):
            continue

        # Collect all images from the few-shot group
        images = get_images(group_path)
        if not images:
            continue

        # Read the corresponding score.txt
        with open(score_path, "r", encoding="utf-8") as f:
            score_text = f.read().strip()

        print(f"Few-shot example from '{group}' in '{category}' with {len(images)} images.")

        fewshot_examples.append({"group": group, "images": images, "score": score_text})

    return fewshot_examples if fewshot_examples else None

def build_prompt(category, group, new_image_paths, fewshot_examples):
    """
    Builds a prompt that includes the few-shot examples (if available) and 
    instructions for evaluating new images.
    """
    prompt_items = []

    # ✅ Add multiple few-shot examples (if available)
    if fewshot_examples:
        print(f"Using few-shot examples from {len(fewshot_examples)} groups in '{category}'")
        for example in fewshot_examples:
            prompt_items.append({"type": "text", "text": f"Example from group '{example['group']}' in '{category}':"})
            for img in example['images']:
                prompt_items.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{encode_image(img)}"}
                })
            prompt_items.append({"type": "text", "text": f"Example persuasion score and explanation:\n{example['score']}"})

    # ✅ Add new images for evaluation
    prompt_items.append({
        "type": "text",
        "text": f"\nNow, evaluate new images in the '{group}' group under the '{category}' category. "
                f"Rank the images **within this group only**, based on their appeal for selling this specific type of product. "
                f"Provide a description, and **persuasion score (1-100)** on each image, along with an explanation for your ranking."
    })

    for img in new_image_paths:
        prompt_items.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{encode_image(img)}"}
        })

    return prompt_items

def process_images():
    """Processes images and saves results in structured category/group folders."""
    
    for category in os.listdir(ROOT_FOLDER):
        category_path = os.path.join(ROOT_FOLDER, category)
        if os.path.isdir(category_path):

            # ✅ Get the two few-shot examples (first two groups inside category folder in `fewshot/`)
            fewshot_examples = load_fewshot_examples(category)
            if not fewshot_examples:
                print(f"No valid few-shot examples found for category '{category}', skipping...")
                continue

            # Process each group within the category.
            for group in os.listdir(category_path):
                group_path = os.path.join(category_path, group)
                if os.path.isdir(group_path):
                    new_image_paths = get_images(group_path)
                    if not new_image_paths:
                        continue  # Skip if no images

                    total_images = sum(len(ex["images"]) for ex in fewshot_examples) + len(new_image_paths)
                    # if total_images >= 7:
                    #     print(f"⏩ Skipping group '{group}' in category '{category}' (total images {total_images} >= 7).")
                    #     continue

                    print(f"Processing group '{group}' in category '{category}' with {len(new_image_paths)} images...")

                    # ✅ Build the prompt
                    prompt_items = build_prompt(category, group, new_image_paths, fewshot_examples)
                    messages = [{"role": "user", "content": prompt_items}]

                    # ✅ Send to Pixtral model
                    try:
                        response = client.chat.complete(model=MODEL_NAME, messages=messages, max_tokens=1024)
                        output_text = response.choices[0].message.content
                    except Exception as e:
                        print(f"Error processing group '{group}' in category '{category}': {e}")
                        output_text = "Error"

                    # ✅ Remove prompt portion and save only model's response
                    cleaned_output_text = output_text.strip()

                    # ✅ Save results in structured folders
                    category_folder = os.path.join(OUTPUT_FOLDER, category)
                    group_folder = os.path.join(category_folder, group)
                    os.makedirs(group_folder, exist_ok=True)

                    txt_filename = os.path.join(group_folder, "output_pixtral_fewshot.txt")

                    with open(txt_filename, "w", encoding="utf-8") as txt_file:
                        txt_file.write(cleaned_output_text)

                    print(f"Cleaned response saved in {txt_filename}")

if __name__ == "__main__":
    process_images()
