"""
NIMA Baseline Scoring — Fixed Version

Runs NIMA inference on all images in the dataset via evaluate_mobilenet.py
and saves raw scores to a CSV.

Key fixes:
- python3 -u flag forces unbuffered stdout so subprocess captures all output
- parse_nima_output uses separate regex for paths and scores (handles blank lines)
- scores parsed from stdout (where evaluate_mobilenet.py actually prints)
- case-insensitive image name matching (evaluate_mobilenet.py lowercases filenames)

Output: nima_baseline_results/nima_scores.csv
        columns: [category, group, image_name, nima_score]
"""

import os
import subprocess
import re
import pandas as pd

# ── Config ─────────────────────────────────────────────────────────────────────
NIMA_SCRIPT   = "/home/debajyoti/neural-image-assessment/evaluate_mobilenet.py"
NIMA_DIR      = "/home/debajyoti/neural-image-assessment"
dataset_image = "/home/debajyoti/paridhi_mtp/nas_final/MTP/Persuasive_Image/Persuasive_Image/home/debajyoti/paridhi_mtp/product_images_real/dataset_image_new"
output_dir    = "./nima_baseline_results"
MAX_IMAGES    = 4
RESIZE        = True
os.makedirs(output_dir, exist_ok=True)

# ── Parse NIMA output ──────────────────────────────────────────────────────────
def parse_nima_output(output_text):
    """
    Robust parser for NIMA output.
    Handles:
    - blank lines
    - '+-' uncertainty
    - variable spacing
    """

    pattern = r"Evaluating\s*:\s*(.*?)\n\s*NIMA Score\s*:\s*([\d\.]+)"

    matches = re.findall(pattern, output_text)

    scores = {}
    for img_path, score in matches:
        filename = os.path.basename(img_path.strip())
        scores[filename] = float(score)

    return scores
# ── Score all images ───────────────────────────────────────────────────────────
results   = []
errors    = []
processed = 0

for category in sorted(os.listdir(dataset_image)):
    category_path = os.path.join(dataset_image, category)
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

        processed += 1
        if processed % 50 == 0:
            print(f"  Progress: {processed} groups, {len(results)} images scored so far")

        # -u forces unbuffered stdout so subprocess.PIPE captures everything
        cmd = ["python3", "-u", NIMA_SCRIPT, "-dir", group_path]
        if RESIZE:
            cmd += ["-resize", "true"]

        try:
            result = subprocess.run(
                cmd,
                cwd=NIMA_DIR,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )

            scores = parse_nima_output(result.stdout)
            # print("\n===== DEBUG OUTPUT =====")
            # print(result.stdout)
            # print("===== STDERR =====")
            # print(result.stderr)

            if not scores:
                print(f"  Warning: no scores parsed for {category}/{group}")
                print(f"  stdout repr: {repr(result.stdout[:400])}")
                errors.append(f"{category}/{group}: no scores parsed")
                continue

            for img_name in images:
                # evaluate_mobilenet.py lowercases filenames internally
                matched_score = scores.get(img_name) or scores.get(img_name.lower())

                if matched_score is None:
                    print(f"  Warning: {img_name} not in NIMA output for {category}/{group}")
                    print(f"  Available keys: {list(scores.keys())}")
                    errors.append(f"{category}/{group}: missing {img_name}")
                    continue

                results.append({
                    "category":   category,
                    "group":      group,
                    "image_name": img_name,
                    "nima_score": matched_score,
                })

        except Exception as e:
            print(f"  Error for {category}/{group}: {e}")
            errors.append(f"{category}/{group}: {e}")

# ── Save ───────────────────────────────────────────────────────────────────────
df = pd.DataFrame(results)
out_path = os.path.join(output_dir, "nima_scores.csv")
df.to_csv(out_path, index=False)

print(f"\nScored {len(df)} images across {df['group'].nunique()} groups")
if len(df) > 0:
    print(f"NIMA score stats: min={df['nima_score'].min():.3f}  "
          f"max={df['nima_score'].max():.3f}  "
          f"mean={df['nima_score'].mean():.3f}  "
          f"std={df['nima_score'].std():.3f}")
print(f"Errors: {len(errors)}")
if errors:
    print("First 10 errors:")
    for e in errors[:10]:
        print(f"  {e}")
print(f"\nSaved to {out_path}")