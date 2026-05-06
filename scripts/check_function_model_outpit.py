import os
import torch
import json
from transformers import AutoProcessor, AutoModelForImageTextToText
from huggingface_hub import login
from qwen_vl_utils import process_vision_info

# Authenticate with Hugging Face
login(token='hf_xxxxxxxxx')

# Define model name
model_name = "Deb123/qwen2.5-vl-7b-pair-finetuned-private"

# Load model and processor
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = AutoProcessor.from_pretrained(model_name, use_fast=False)
model = AutoModelForImageTextToText.from_pretrained(model_name).to(device)

print("Model and processor loaded successfully!")

## SET PATHS
# Define root folder where images are stored
root_folder = "debajyoti/selected_product_images"

# Store results
results = {}

# Iterate over product categories (TV, Laptop, etc.)
for category in os.listdir(root_folder):
    category_path = os.path.join(root_folder, category)
    if os.path.isdir(category_path):
        results[category] = {}

        # Iterate over groups within each category
        for group in os.listdir(category_path):
            group_path = os.path.join(category_path, group)
            if os.path.isdir(group_path):
                images = []
                image_files = sorted(os.listdir(group_path))  # Ensures sorted order of images

                # Collect images in sorted order
                for image_file in image_files:
                    image_path = os.path.join(group_path, image_file)
                    images.append({"type": "image", "image": f"file://{image_path}"})

                if not images:
                    continue

                messages = [
                    {
                        "role": "user",
                        "content": images + [
                            {
                                "type": "text",
                                "text": f"You are evaluating images in the '{group}' group under the '{category}' product category. "
                                        f"Rank the images, based on their appeal for selling '{category}' product. "
                                        f"Provide description , and **persuasion score (1-100)** for each image and explain the ranking."
                            }
                        ],
                    }
                ]

                # Prepare inputs
                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                ).to(device)

                # Inference
                with torch.no_grad():
                    generated_ids = model.generate(**inputs, max_new_tokens=800)

                # Extract output text
                output_text = processor.batch_decode(
                    generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0]

                # Parse persuasion scores from model output (enhanced parsing)
                scores = []
                for idx, image_file in enumerate(image_files):
                    default_score = 50
                    try:
                        score_line = [line for line in output_text.split('\n') if image_file in line][0]
                        score = int(next((word for word in score.split() if word.isdigit()), 50))
                    except (ValueError, IndexError):
                        score = 50
                    scores.append((image_file, score))

                # Sort based on persuasion scores
                ranked_images = sorted(scores, key=lambda x: x[1], reverse=True)

                # Store ranked results with explanations
                results[category][group] = [
                    {"image": img, "score": score, "reasoning": output_text}
                    for img, score in ranked_images
                ]

# Save results to JSON file
with open("output5.json", "w", encoding="utf-8") as json_file:
    json.dump(results, json_file, indent=4, ensure_ascii=False)

print("Results saved in output4.json")

# Save human-readable TXT file
with open("output1.txt", "w", encoding="utf-8") as txt_file:
    for category, groups in results.items():
        txt_file.write(f"\n📌 **Category: {category}**\n")
        for group, images in groups.items():
            txt_file.write(f"\n🔹 **Group: {group}**\n")
            for rank, img_data in enumerate(images, 1):
                # txt_file.write(f"    🏆 Rank {rank}: {img_data['image']} | 🎯 Score: {img_data['score']}/100\n")
                txt_file.write(f"{img_data['reasoning']}\n\n")

print("Results saved in output4.txt")