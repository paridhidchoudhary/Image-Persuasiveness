import os
import random
from datasets import load_from_disk
from PIL import Image
import numpy as np

import io

## SET PATHS
# Load the saved filtered dataset
ds = load_from_disk("home/debajyoti/paridhi/subdataset_new")

# Define base directories
output_dir = "home/debajyoti/paridhi_mtp/product_images_real/dataset_image_new"
os.makedirs(output_dir, exist_ok=True)

group_dir = "home/debajyoti/paridhi_mtp/product_images_real/groups"
product_dir = "home/debajyoti/paridhi_mtp/product_images_real/products"

# Select products for comparison
target_products = ["laptop", "tv", "cell phone", "backpack", "toaster", "suitcase", "handbag", "refrigerator", "microwave","mouse","keyboard",
"remote","oven","bottle","cup","fork","knife","spoon","chair","couch"]

#target_products = ["laptop", "tv", "cell phone", "backpack", "toaster", "suitcase", "handbag", "refrigerator", "microwave"]
selected_groups = {}

# Select two groups per product
for product in target_products:
    product_path = os.path.join(product_dir, product)
    if os.path.exists(product_path) and os.path.isdir(product_path):
        groups = [name for name in os.listdir(product_path) if os.path.isdir(os.path.join(product_path, name))]
        selected_groups[product] = random.sample(groups, 50) if len(groups) >= 50 else groups

# Function to fetch image by ID (Optimized)
# def fetch_image(image_id):
#     for entry in ds:
#         if entry["image_id"] == image_id:
#             image = entry["image"]
#             if isinstance(image, dict) and "bytes" in image:
#                 image = Image.open(io.BytesIO(image["bytes"]))  # Convert bytes to PIL Image
#             return image
#     return None

image_lookup = {entry["image_id"]: entry["image"] for entry in ds}

def fetch_image(image_id):
    image = image_lookup.get(image_id)
    if image and isinstance(image, dict) and "bytes" in image:
        image = Image.open(io.BytesIO(image["bytes"]))  # Convert bytes to PIL Image
    return image


# Fetch and save images from selected groups
for product, groups in selected_groups.items():
    product_output_dir = os.path.join(output_dir, product)
    os.makedirs(product_output_dir, exist_ok=True)

    for group in groups:
        metadata_path = os.path.join(group_dir, group, "metadata.txt")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                lines = f.readlines()
            
            original_id = None
            edit_ids = []
            for line in lines:
                if line.startswith("Original:"):
                    original_id = line.split(":")[1].strip()
                elif line.startswith("Edits:"):
                    edit_ids = line.split(":")[1].strip().split(", ") if line.split(":")[1].strip() else []

            # Create a folder for each group
            group_output_dir = os.path.join(product_output_dir, group)
            os.makedirs(group_output_dir, exist_ok=True)

            # Save original image
            try:
                original_image = fetch_image(original_id)
                if original_image:
                    original_image_path = os.path.join(group_output_dir, f"original_{original_id}.jpg")
                    original_image.save(original_image_path)
            except Exception as e:
                print(f"Error saving original image {original_id}: {e}")

            # Save all edited images
            for idx, edit_id in enumerate(edit_ids):
                try:
                    edited_image = fetch_image(edit_id)
                    if edited_image:
                        edited_image_path = os.path.join(group_output_dir, f"edit_{idx+1}_{edit_id}.jpg")
                        edited_image.save(edited_image_path)
                except Exception as e:
                    print(f"Error saving edited image {edit_id}: {e}")

print("Images have been saved successfully.")