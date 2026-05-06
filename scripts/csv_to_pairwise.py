import pandas as pd
from itertools import combinations
import os

# --- PATHS ---
input_csv = "datasets/persuasion/persuasion_clip.csv"  # your CSV
output_rank_csv = "datasets/persuasion/persuasion_clip_rank.csv"

# --- REMOVE OLD FILE IF EXISTS ---
if os.path.exists(output_rank_csv):
    os.remove(output_rank_csv)
    print(f"Old file removed: {output_rank_csv}")

# --- READ DATA ---
df = pd.read_csv(input_csv)

# --- INIT PAIRS LIST ---
pairs = []

# --- GROUP BY CATEGORY + GROUP ---
grouped = df.groupby(["cname", "group"])

for (cname, group_name), group_df in grouped:
    # Only keep images with valid score
    group_df = group_df.dropna(subset=["value"])
    
    # Generate all pair combinations within the group
    for img1, img2 in combinations(group_df.to_dict("records"), 2):
        # Decide which image is more persuasive
        if img1["value"] == img2["value"]:
            continue  # skip ties (optional)
        if img1["value"] > img2["value"]:
            winner, loser = img1, img2
        else:
            winner, loser = img2, img1

        pairs.append({
            "image_a": winner["filepath"],
            "image_b": loser["filepath"],
            "label": 1  # 1 = image_a more persuasive than image_b
        })

# --- SAVE PAIRWISE CSV ---
rank_df = pd.DataFrame(pairs)
rank_df.to_csv(output_rank_csv, index=False)
print(f"Pairwise ranking CSV saved: {output_rank_csv} with {len(rank_df)} pairs")
