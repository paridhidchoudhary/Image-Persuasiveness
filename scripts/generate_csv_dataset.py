import os
import pandas as pd
from simple_data_preprocess import extract_text_data  

# --- PATHS ---

image_root = "/home/debajyoti/paridhi_mtp/MTP/Persuasive_Image/Persuasive_Image/home/debajyoti/paridhi_mtp/product_images_real/dataset_image_new"
response_root = "/home/debajyoti/paridhi_mtp/MTP/Persuasive_Image/Persuasive_Image/home/debajyoti/paridhi_mtp/product_images_real/final_data"
output_csv = "datasets/persuasion/persuasion_clip.csv"

rows = []

# --- MAIN LOOP ---
for category in os.listdir(image_root):
    category_path = os.path.join(image_root, category)
    if not os.path.isdir(category_path):
        continue

    for group in os.listdir(category_path):
        group_path = os.path.join(category_path, group)
        if not os.path.isdir(group_path):
            continue

        resp_path = os.path.join(response_root, category, group, "user_output.txt")
        if not os.path.exists(resp_path):
            continue

        # extract persuasion scores using your function
        try:
            extracted_info, ranking = extract_text_data(resp_path)
        except Exception as e:
            print(f"Error extracting {category}/{group}: {e}")
            continue

        if not extracted_info:
            continue

        # Get all valid image files in the same group
        image_files = sorted([
            f for f in os.listdir(group_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
        ])

        # Map Image N → actual filename (1-indexed in output)
        for info in extracted_info:
            idx = info["image_num"] - 1
            if idx < len(image_files):
                image_name = image_files[idx]
                image_path = os.path.join(group_path, image_name)
                rows.append({
                    "filepath": image_path,
                    "title": f"Rank the persuasiveness of this {category} image.",
                    "cname": category,
                    "group": group,
                    "value": info["score"],
                    "sample_weights_by_class": 1.0
                })
            else:
                print(f"Missing image index {info['image_num']} in {category}/{group}")

# --- SAVE CSV (OVERWRITE) ---
df = pd.DataFrame(rows)

if os.path.exists(output_csv):
    os.remove(output_csv)
    print(f"Old CSV removed: {output_csv}")

df.to_csv(output_csv, index=False)
print(f"New CSV saved: {output_csv} with {len(df)} entries")

# --- OPTIONAL SANITY CHECK ---
if len(df) > 0:
    print("\nSample entries:")
    print(df.sample(min(3, len(df))))
