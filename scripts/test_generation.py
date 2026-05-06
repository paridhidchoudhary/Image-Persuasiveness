import os
import torch
import random
import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from huggingface_hub import login
from qwen_vl_utils import process_vision_info
from sklearn.model_selection import train_test_split

from simple_data_preprocess import extract_text_data

SEED = 48
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

login(token="hf_xxxxxxxxxxxxxxxxxxx")

data_root = "home/debajyoti/paridhi_mtp/product_images_real"
dataset_image = os.path.join(data_root, "dataset_image_new")
dataset_user_preferred = os.path.join(data_root, "final_data")
FINETUNED_MODEL_NAME = "vlm_finetuned_listwise"
MAX_IMAGES = 4

# ── Load dataset (same as training) ──────────────────────────
data = []
for category in os.listdir(dataset_image):
    cat_path = os.path.join(dataset_image, category)
    if not os.path.isdir(cat_path):
        continue
    for group in os.listdir(cat_path):
        group_path = os.path.join(cat_path, group)
        if not os.path.isdir(group_path):
            continue
        images = sorted([
            os.path.join(group_path, img)
            for img in os.listdir(group_path)
            if img.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
        ])
        if len(images) > MAX_IMAGES or len(images) == 0:
            continue
        gt_path = os.path.join(dataset_user_preferred, category, group, "user_output.txt")
        if os.path.exists(gt_path):
            data.append({
                "images": images,
                "category": category,
                "group": group,
                "ground_truth_path": gt_path
            })

# Same split as training
_, test_data = train_test_split(data, test_size=0.05, random_state=48)
print(f"Test set size: {len(test_data)}")
print(f"Using first 3 samples:\n")

# ── Load model ────────────────────────────────────────────────
print(f"Loading {FINETUNED_MODEL_NAME}...")
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = AutoProcessor.from_pretrained(FINETUNED_MODEL_NAME, use_fast=False)
model = AutoModelForImageTextToText.from_pretrained(
    FINETUNED_MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)
model.eval()
print(f"Model loaded on {device}\n")
print("=" * 70)

# ── Generate for first 3 test samples ────────────────────────
for i, sample in enumerate(test_data[:3]):
    category = sample["category"]
    group = sample["group"]
    images = sample["images"]
    gt_path = sample["ground_truth_path"]

    print(f"\n[Sample {i+1}] {category}/{group}  ({len(images)} images)")
    print("-" * 70)

    # GT scores
    extracted_info, _ = extract_text_data(gt_path)
    if extracted_info:
        extracted_info.sort(key=lambda x: x["image_num"])
        gt_scores = [item.get("score", 0) for item in extracted_info]
        print(f"GT scores (raw): {gt_scores}")
    else:
        print("GT: could not extract")

    # Build prompt — identical to training collate_fn
    sorted_images = []
    for img_path in images:
        img = Image.open(img_path).convert("RGB")
        if img.width > 256 or img.height > 256:
            factor = 256 / float(max(img.width, img.height))
            img = img.resize((int(img.width * factor), int(img.height * factor)), Image.LANCZOS)
        sorted_images.append({"type": "image", "image": img})

    messages = [{
        "role": "user",
        "content": sorted_images + [{
            "type": "text",
            "text": f"You are evaluating images in the '{group}' group under the '{category}' product category. "
                    f"Rank the images, based on their appeal for selling '{category}' product. "
                    f"Provide description, and *persuasion score (1-100)* for each image and explain the ranking."
        }]
    }]

    prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=prompt_text,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        truncation=True,
        max_length=2048,
        return_tensors="pt"
    ).to(device)

    # Generate text output
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,        # greedy — deterministic
            temperature=1.0,
            pad_token_id=processor.tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens
    input_len = inputs["input_ids"].shape[1]
    generated_text = processor.tokenizer.decode(
        generated_ids[0][input_len:],
        skip_special_tokens=True
    )

    print(f"\nGenerated output:")
    print(generated_text)

    # Try to parse scores from generated text
    print(f"\nAttempting score extraction from generated text...")
    # Save to temp file and use extract_text_data
    tmp_path = f"/tmp/gen_output_{i}.txt"
    with open(tmp_path, "w") as f:
        f.write(generated_text)

    try:
        parsed_info, parsed_ranking = extract_text_data(tmp_path)
        if parsed_info:
            parsed_info.sort(key=lambda x: x["image_num"])
            parsed_scores = [item.get("score", None) for item in parsed_info]
            print(f"Parsed scores: {parsed_scores}")
            print(f"Parsed ranking: {parsed_ranking}")
        else:
            print("Could not parse scores from generated text")
    except Exception as e:
        print(f"Parsing error: {e}")

    print("=" * 70)

print("\nDone. Key things to check:")
print("1. Does the generated text contain scores in 1-100 range?")
print("2. Are the parsed scores matching what's in the text?")
print("3. Does the ranking look reasonable compared to GT?")