import os
import base64
import re
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
# ✅ Initialize API Key and Model
API_KEY = "MgFASHu9FVkLBk2QRPqTL70trCY0QqoZ"
MODEL_NAME = "pixtral-large-latest"

## SET PATHS
ROOT_FOLDER = "home/debajyoti/paridhi_mtp/product_images_real/dataset_image_new"
#ROOT_FOLDER = "/home/deepg/NAS/Downloads/MTP-2-persuasion/dataset/dataset_image copy"  # Root image folder
OUTPUT_ROOT = "home/debajyoti/paridhi_mtp/product_images_real/dataset_response_new"
#OUTPUT_ROOT = "/home/deepg/NAS/Downloads/MTP-2-persuasion/dataset/dataset_response copy" # Output folder for responses

# ✅ Initialize Mistral client
client = Mistral(api_key=API_KEY)

def encode_image(image_path):
    """Encodes an image as a base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def get_images(folder_path):
    """Fetch all image paths in a folder, sorted lexicographically."""
    return sorted(
        [os.path.join(folder_path, file)
        for file in os.listdir(folder_path)
        if file.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))]
    )


def evaluate_persuasiveness(category, group, image_paths):
    """Sends multiple images to Pixtral Large for analysis."""
    encoded_images = [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image(img)}"}}
        for img in image_paths
    ]

    prompt_text = (
        f"You are evaluating images in the '{group}' group under the '{category}' product category. "
        f"Rank the images **within this group only**, based on their appeal for selling this specific type of product. "
        f"Provide a description, and **persuasion score (1-100)** for each image and explain the ranking."
    )

    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": prompt_text}] + encoded_images
        }
    ]

    try:
        response = client.chat.complete(model=MODEL_NAME, messages=messages, max_tokens=1024)
        return response.choices[0].message.content  # Extract the text response
    except Exception as e:
        print(f"Error processing images: {e}")
        return "Error"

def clean_response(response_text, prompt_text):
    """Removes the initial prompt from the response if present."""
    cleaned_text = response_text.strip()
    
    # If response starts with the prompt, remove it
    if cleaned_text.startswith(prompt_text):
        cleaned_text = cleaned_text[len(prompt_text):].strip()
    
    # Remove extra whitespace or newlines at the start
    cleaned_text = re.sub(r"^\s*\n+", "", cleaned_text)
    
    return cleaned_text

def process_images():
    """Processes images and saves results in separate category/group folders."""
    
    # Iterate over product categories (TV, Laptop, etc.)
    for category in os.listdir(ROOT_FOLDER):
        category_path = os.path.join(ROOT_FOLDER, category)
        if os.path.isdir(category_path):

            # Iterate over groups within each category
            for group in os.listdir(category_path):
                group_path = os.path.join(category_path, group)
                if os.path.isdir(group_path):
                    image_paths = get_images(group_path)

                    if not image_paths:
                        print(f"No images found in {group} ({category}). Skipping...")
                        continue  

                    print(f"Processing {len(image_paths)} images in {group} ({category})...")

                    # Generate the prompt text for comparison
                    prompt_text = (
                        f"You are evaluating images in the '{group}' group under the '{category}' product category. "
                        f"Rank the images **within this group only**, based on their appeal for selling this specific type of product. "
                        f"Provide a description, and **persuasion score (1-100)** for each image and explain the ranking."
                    )

                    # Get Pixtral evaluation response
                    response_text = evaluate_persuasiveness(category, group, image_paths)

                    # Clean response by removing the prompt
                    cleaned_output_text = clean_response(response_text, prompt_text)

                    # Define output folder structure matching input
                    category_folder = os.path.join(OUTPUT_ROOT, category)
                    group_folder = os.path.join(category_folder, group)

                    # Create folders if they don’t exist
                    os.makedirs(group_folder, exist_ok=True)

                    # Define TXT output file path
                    txt_filename = os.path.join(group_folder, "output_pixtral_zeroshot.txt")

                    # Save only the cleaned assistant response as TXT
                    with open(txt_filename, "w", encoding="utf-8") as txt_file:
                        txt_file.write(cleaned_output_text)

                    print(f"Cleaned response saved in {txt_filename}")

if __name__ == "__main__":
    process_images()
