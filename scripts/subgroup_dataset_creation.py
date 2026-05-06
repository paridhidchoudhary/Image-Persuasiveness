import os
import pandas as pd
from datasets import load_dataset, Dataset, Features, Value, Image, concatenate_datasets
from PIL import Image as PILImage
import io

# Load the dataset
ds = load_dataset("vismin_dataset")
import random 
import torch 
import numpy as np
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

## SET PATHS
group_dir = "home/debajyoti/paridhi_mtp/product_images_real/products"
# output_file = "home/debajyoti/paridhi/subdataset"
output_file = "home/debajyoti/paridhi/subdataset_new"

# Target products
target_products = ["laptop", "tv", "cell phone", "backpack", "toaster", "suitcase", "handbag", "refrigerator", "microwave","mouse","keyboard",
"remote","oven","bottle","cup","fork","knife","spoon","chair","couch"]
#target_products = ["laptop", "tv", "cell phone", "backpack", "toaster", "suitcase", "handbag", "refrigerator", "microwave"]

# Dictionary to store image mappings
image_dict = {}

# Extract image_id and edit_ids from metadata
for product in target_products:
    product_path = os.path.join(group_dir, product)
    if os.path.exists(product_path) and os.path.isdir(product_path):
        groups = [name for name in os.listdir(product_path) if os.path.isdir(os.path.join(product_path, name))]
        
        for group in groups:
            metadata_path = os.path.join(product_path, group, "metadata.txt")
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
                
                if original_id:
                    image_dict[original_id] = edit_ids
                    for edit_id in edit_ids:
                        image_dict[edit_id] = []  # Ensure all edits are included in lookup

# Create a new dataset by filtering relevant entries
data = []
for entry in ds["train"]:
    if entry["image_id"] in image_dict:
        # Convert image to Hugging Face `Image` format
        if isinstance(entry["image"], PILImage.Image):  # Ensure image is in PIL format
            img_byte_arr = io.BytesIO()
            entry["image"].save(img_byte_arr, format="JPEG")  # Convert to bytes
            entry["image"] = {"bytes": img_byte_arr.getvalue()}  # Store as bytes
        
        data.append(entry)

# Debugging: Print count of filtered entries
print(f"Number of matched entries: {len(data)}")

# If no data found, exit to prevent errors
if len(data) == 0:
    print("No matching data found. Exiting without saving.")
    exit()

# Convert to a Pandas DataFrame
df = pd.DataFrame(data)

# Extract column names and define features dynamically
columns = ds["train"].column_names
features_dict = {col: Value("string") for col in columns}  # Default to string

# Handle special types
if "image" in columns:
    features_dict["image"] = Image(decode=True)  # Ensure images are stored in the correct format

# Define features
features = Features(features_dict)

# Process in smaller batches to avoid memory issues
batch_size = 1000  # Adjust based on available RAM
dataset_splits = []

for start in range(0, len(df), batch_size):
    batch_df = df.iloc[start:start + batch_size]
    batch_dataset = Dataset.from_pandas(batch_df, features=features)
    dataset_splits.append(batch_dataset)

# Concatenate all dataset splits properly
final_dataset = concatenate_datasets(dataset_splits)

# Save dataset in Hugging Face format
final_dataset.save_to_disk(output_file)

print(f"Filtered dataset saved at: {output_file}")
