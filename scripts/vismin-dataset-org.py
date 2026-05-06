import os
from datasets import load_dataset

# Load the dataset
#ds = load_dataset("mair-lab/vismin")
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
# Define base directory for groups and products
## SET PATHS
base_dir = "home/debajyoti/paridhi_mtp/product_images_real"
group_dir = os.path.join(base_dir, "groups")
os.makedirs(group_dir, exist_ok=True)

# Initialize a dictionary to hold grouped images and their captions
grouped_images = {}

# List of products to track
products = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

# First pass: Create groups and track products
for entry in ds['train']:
    image_id = entry['image_id']
    source_image_id = entry['source_image_id']
    caption = entry['caption'] if entry['caption'] else ''  # Ensure caption is not None
    
    if not source_image_id :  # Original image
        if image_id not in grouped_images:
            grouped_images[image_id] = {
                'original': image_id,
                'edits': [],
                'products': set(),
                'caption': caption
            }
        else:
            # Always update the caption
            grouped_images[image_id]['caption'] = caption  

        # Extract products from caption
        lower_caption = caption.lower()
        for product in products:
            if product in lower_caption:
                grouped_images[image_id]['products'].add(product)

    else:  # Edited image
        if source_image_id not in grouped_images:
            grouped_images[source_image_id] = {
                'original': source_image_id,
                'edits': [],
                'products': set(),
                'caption': caption  # Assign caption here
            }
        grouped_images[source_image_id]['edits'].append(image_id)

# Second pass: Handle products for edited images
for group_id, group_data in grouped_images.items():
    #print(2)
    if group_data['caption']:
        #print(3)
        lower_caption = group_data['caption'].lower()
        for product in products:
            if product in lower_caption:
                group_data['products'].add(product)
                #print(1)

# Create group folders and product associations
for group_id, group_data in grouped_images.items():
    # Create group folder
    group_folder = os.path.join(group_dir, f"group_{group_id}")
    os.makedirs(group_folder, exist_ok=True)
    
    # Save group metadata
    with open(os.path.join(group_folder, 'metadata.txt'), 'w') as f:
        f.write(f"Original: {group_data['original']}\n")
        f.write(f"Edits: {', '.join(group_data['edits'])}\n")
        f.write(f"Caption: {group_data['caption']}\n")
        f.write(f"Products: {', '.join(group_data['products'])}\n")
    
    # Create product folders and symlinks
    for product in group_data['products']:
        product_dir = os.path.join(base_dir, 'products', product)
        os.makedirs(product_dir, exist_ok=True)
        
        # Create symlink to group folder
        symlink_path = os.path.join(product_dir, f"group_{group_id}")
        if not os.path.exists(symlink_path):
            os.symlink(os.path.abspath(group_folder), symlink_path)

print("Organization complete!")
