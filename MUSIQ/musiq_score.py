"""
MUSIQ Baseline Scoring — Fixed Version

Runs MUSIQ inference on all images in the dataset and saves raw scores.
No normalization needed — ranking metrics (Kendall, Spearman, Agreement,
Top Accuracy) are invariant to monotone score transformations, so raw
MUSIQ scores work just as well as normalized ones.

Output: musiq_scores.csv with columns [category, group, image_path, musiq_score]
        image_path is stored as basename only to avoid path-matching issues.
"""

import torch
import pyiqa
from PIL import Image
import pandas as pd
import numpy as np
from tqdm import tqdm
import os

# ── Config ─────────────────────────────────────────────────────────────────────
dataset_image_path = "/home/debajyoti/paridhi_mtp/nas_final/MTP/Persuasive_Image/Persuasive_Image/home/debajyoti/paridhi_mtp/product_images_real/dataset_image_new"
output_dir         = "./musiq_baseline_results_2"
MAX_IMAGES         = 4
os.makedirs(output_dir, exist_ok=True)

# ── Load MUSIQ ─────────────────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading MUSIQ on {device}...")
model = pyiqa.create_metric('musiq', device=device)
print("MUSIQ loaded.")

# ── Score all images — single pass ────────────────────────────────────────────
results = []

for category in sorted(os.listdir(dataset_image_path)):
    category_path = os.path.join(dataset_image_path, category)
    if not os.path.isdir(category_path):
        continue

    for group in sorted(os.listdir(category_path)):
        group_path = os.path.join(category_path, group)
        if not os.path.isdir(group_path):
            continue

        images = sorted([
            f for f in os.listdir(group_path)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
        ])

        if len(images) > MAX_IMAGES or len(images) == 0:
            continue

        for img_name in images:
            img_full_path = os.path.join(group_path, img_name)
            try:
                score = model(img_full_path).item()
                results.append({
                    "category":   category,
                    "group":      group,
                    "image_name": img_name,          # basename only — avoids path mismatch
                    "image_path": img_full_path,     # full path for reference
                    "musiq_score": float(score),
                })
            except Exception as e:
                print(f"  Error on {img_full_path}: {e}")

df = pd.DataFrame(results)

# Print stats for reference
print(f"\nScored {len(df)} images across {df['group'].nunique()} groups")
print(f"Raw MUSIQ score stats:")
print(f"  min={df['musiq_score'].min():.2f}  max={df['musiq_score'].max():.2f}"
      f"  mean={df['musiq_score'].mean():.2f}  std={df['musiq_score'].std():.2f}")

out_path = os.path.join(output_dir, "musiq_scores.csv")
df.to_csv(out_path, index=False)
print(f"\nSaved to {out_path}")