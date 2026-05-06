import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import unsloth
from unsloth import FastVisionModel
import json
import torch
import gc
import re
from PIL import Image
from sklearn.model_selection import train_test_split
from transformers import AutoProcessor, TrainingArguments, Trainer, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType
from huggingface_hub import login, HfApi
from qwen_vl_utils import process_vision_info
from itertools import combinations
import random 
import numpy as np
import torch.nn as nn

torch._dynamo.config.suppress_errors = True

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

login(token="hf_xxxxxxxxxxxxxxxxxxxxxx")
repo_id = "Deb123/qwen2.5-vl-7b-pair-new-code-finetuned-private"
api = HfApi()
api.create_repo(repo_id, private=True, exist_ok=True)

data_root = "home/debajyoti/paridhi_mtp/product_images_real"
dataset_image = os.path.join(data_root, "dataset_image_new")
dataset_response = os.path.join(data_root, "final_data")
output_model = "./vlm_finetuned_pairwise"
debug_output_dir = "./debug_output"
os.makedirs(debug_output_dir, exist_ok=True)

model_name = "unsloth/Qwen2.5-VL-7B-Instruct"
processor = AutoProcessor.from_pretrained(model_name, use_fast=False)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

torch.cuda.empty_cache()
gc.collect()

# Load with proper memory constraints - no invalid parameters
model, tokenizer = FastVisionModel.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    use_gradient_checkpointing="unsloth", 
    device_map="balanced",  # Use balanced instead of auto
    max_seq_length=2048,
    low_cpu_mem_usage=True,  # Reduce CPU memory usage during loading
)

if hasattr(model.config, 'use_fused_cross_entropy'):
    model.config.use_fused_cross_entropy = False

model.config.use_cache = False

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=4,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
)
model = get_peft_model(model, peft_config)

torch.cuda.empty_cache()
gc.collect()

from simple_data_preprocess import extract_text_data, find_ranking_position, extract_ranking, extract_score

def resize_if_needed(img, max_size=256):
    if img.width > max_size or img.height > max_size:
        scaling_factor = max_size / float(max(img.width, img.height))
        new_width = int(img.width * scaling_factor)
        new_height = int(img.height * scaling_factor)
        return img.resize((new_width, new_height), Image.LANCZOS)
    return img

def create_pair_data(images, indices, extracted_info, category, pairwise_data, original_ranking=None, debug_info=None):
    """Create a data entry for a pair of images with reindexed info."""
    pair_images = [images[i] for i in indices]
    
    pair_scores = []
    pair_extracted_info = {}
    
    ordinal_to_num = {"First": 1, "Second": 2, "Third": 3, "Fourth": 4, "Fifth": 5}
    num_to_ordinal = {1: "First", 2: "Second", 3: "Third", 4: "Fourth", 5: "Fifth"}
    
    for new_idx, original_idx in enumerate(indices):
        if original_idx < len(extracted_info):
            img_info = extracted_info[original_idx]
            score = img_info.get("score", 0)
            pair_scores.append((new_idx, score))
            
            new_key = f"Image {new_idx + 1}"
            full_text = img_info.get("full_text", "")
            
            updated_text = full_text
            
            if re.search(r'^\d+\.', updated_text, re.MULTILINE):
                updated_text = re.sub(r'^(\d+)\.', f"{new_idx + 1}.", updated_text, flags=re.MULTILINE)
            
            image_numbers = set(re.findall(r'Image\s+(\d+)', updated_text))
            
            for img_num in image_numbers:
                updated_text = re.sub(
                    r'Image\s+' + re.escape(img_num), 
                    f"Image {new_idx + 1}", 
                    updated_text
                )
            
            ordinal_pattern = r'(First|Second|Third|Fourth|Fifth)'
            if re.search(ordinal_pattern, updated_text):
                for ordinal, num in ordinal_to_num.items():
                    updated_text = re.sub(
                        ordinal,
                        num_to_ordinal[new_idx + 1],
                        updated_text
                    )
            
            pair_extracted_info[new_key] = {
                "full_text": updated_text,
                "score": score
            }
    
    if len(indices) == 2 and original_ranking:
        ranking_text = original_ranking
    else:
        pair_scores.sort(key=lambda x: x[1], reverse=True)
        
        ranking_text = "###Ranking:\n"
        for rank, (new_idx, score) in enumerate(pair_scores, 1):
            ranking_text += f"{rank}. **Image {new_idx + 1}** - Persuasion Score: {score}/100\n"
    
    pair_data = {
        "images": pair_images,
        "extracted_info": pair_extracted_info,
        "ranking": ranking_text,
        "category": category,
        "group": os.path.basename(os.path.dirname(pair_images[0]))
    }
    
    pairwise_data.append(pair_data)
    
    if debug_info is not None:
        debug_info["pairs"].append(pair_data)
    
    return pair_data

# --- Create pairwise dataset ---
pairwise_data = []
debug_examples = {
    "3_image_examples": [],
    "4_image_examples": [],
    "2_image_examples": []
}

for category in os.listdir(dataset_image):
    category_path = os.path.join(dataset_image, category)
    if os.path.isdir(category_path):
        for group in os.listdir(category_path):
            group_path = os.path.join(dataset_image, category, group)
            if os.path.isdir(group_path):
                images = sorted([
                    os.path.join(group_path, img)
                    for img in os.listdir(group_path)
                    if img.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
                ])
                
                response_path = os.path.join(dataset_response, category, group, "user_output.txt")
                if os.path.exists(response_path):
                    try:
                        extracted_info, ranking = extract_text_data(response_path)
                        
                        if not extracted_info or not isinstance(extracted_info, list):
                            print(f"Warning: Invalid extracted_info for {response_path}")
                            continue
                            
                    except Exception as e:
                        print(f"Error processing {response_path}: {e}")
                        continue
                    
                    original_idx = None
                    for idx, img_path in enumerate(images):
                        if "original" in img_path.lower():
                            original_idx = idx
                            break
                    
                    if original_idx is None and len(images) > 0:
                        original_idx = len(images) - 1
                    
                    debug_info = None
                    
                    if len(images) == 2 and len(debug_examples["2_image_examples"]) < 10:
                        debug_info = {
                            "category": category,
                            "group": group,
                            "original_images": images,
                            "original_info": extracted_info,
                            "original_ranking": ranking,
                            "original_idx": original_idx,
                            "pairs": []
                        }
                        debug_examples["2_image_examples"].append(debug_info)

                    if len(images) == 3 and len(debug_examples["3_image_examples"]) < 30:
                        debug_info = {
                            "category": category,
                            "group": group,
                            "original_images": images,
                            "original_info": extracted_info,
                            "original_ranking": ranking,
                            "original_idx": original_idx,
                            "pairs": []
                        }
                        debug_examples["3_image_examples"].append(debug_info)
                    
                    if len(images) == 4 and len(debug_examples["4_image_examples"]) < 30:
                        debug_info = {
                            "category": category,
                            "group": group,
                            "original_images": images,
                            "original_info": extracted_info,
                            "original_ranking": ranking,
                            "original_idx": original_idx,
                            "pairs": []
                        }
                        debug_examples["4_image_examples"].append(debug_info)
                    
                    if original_idx is not None and len(extracted_info) > 0:
                        if len(images) > 2:
                            edited_indices = [i for i in range(len(images)) if i != original_idx]
                            
                            if len(images) == 3:
                                for edit_idx in edited_indices:
                                    pair_indices = [edit_idx, original_idx]
                                    create_pair_data(images, pair_indices, extracted_info, category, pairwise_data, debug_info=debug_info)
                            
                            elif len(images) == 4:
                                if len(edited_indices) >= 2:
                                    create_pair_data(images, edited_indices[:2], extracted_info, category, pairwise_data, debug_info=debug_info)
                                
                                if len(edited_indices) >= 1:
                                    create_pair_data(images, [edited_indices[-1], original_idx], extracted_info, category, pairwise_data, debug_info=debug_info)
                        
                        elif len(images) == 2:
                            create_pair_data(images, [0, 1], extracted_info, category, pairwise_data, original_ranking=ranking, debug_info=debug_info)

with open(os.path.join(debug_output_dir, "3_image_examples.json"), "w") as f:
    json.dump(debug_examples["3_image_examples"], f, indent=2)

with open(os.path.join(debug_output_dir, "4_image_examples.json"), "w") as f:
    json.dump(debug_examples["4_image_examples"], f, indent=2)

with open(os.path.join(debug_output_dir, "2_image_examples.json"), "w") as f:
    json.dump(debug_examples["2_image_examples"], f, indent=2)

print(f"Created {len(pairwise_data)} pairwise training samples")
print(f"Debug examples saved to {debug_output_dir}")

train_data, test_data = train_test_split(pairwise_data, test_size=0.05, random_state=48)
train_data, val_data = train_test_split(train_data, test_size=0.05, random_state=48)
print(f"Training samples: {len(train_data)}, Validation samples: {len(val_data)}")

def collate_fn(batch):
    """Fixed collate function with num_items_in_batch handling"""
    batch = batch[0]
    try:
        images = []
        for img_path in batch["images"]:
            img = resize_if_needed(Image.open(img_path).convert("RGB"))
            images.append(img)
        
        category = batch["category"]
        group = batch.get("group", "unknown")
        
        image_content = [{"type": "image", "image": img} for img in images]
        
        user_prompt = f"You are evaluating images in the '{group}' group under the '{category}' product category. " \
                      f"Rank the images, based on their appeal for selling '{category}' product. " \
                      f"Provide description, and **persuasion score (1-100)** for each image and explain the ranking."
        
        user_message = {
            "role": "user",
            "content": image_content + [{"type": "text", "text": user_prompt}]
        }
        
        extracted_info_text = ""
        for key, info in batch["extracted_info"].items():
            extracted_info_text += f"{key}:\n{info['full_text']}\n\n"
        
        assistant_message = {
            "role": "assistant",
            "content": extracted_info_text + "\n\n" + batch["ranking"]
        }
        
        prompt_text = processor.apply_chat_template([user_message], tokenize=False, add_generation_prompt=True)
        full_text = processor.apply_chat_template([user_message, assistant_message], tokenize=False, add_generation_prompt=False)
        
        image_inputs, video_inputs = process_vision_info([user_message])
        
        inputs = processor(
            text=prompt_text,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            truncation=True,
            max_length=2048,
            return_tensors="pt"
        )
        
        labels = processor(
            text=full_text,
            padding=True,
            truncation=True,
            max_length=2048,
            return_tensors="pt"
        )["input_ids"]
        
        input_len = inputs["input_ids"].shape[1]
        labels_len = labels.shape[1]
        
        if labels_len < input_len:
            padding = torch.full((1, input_len - labels_len), -100, dtype=torch.long)
            labels = torch.cat([labels, padding], dim=1)
        elif labels_len > input_len:
            labels = labels[:, :input_len]
        
        inputs["labels"] = torch.where(
            inputs["input_ids"] == processor.tokenizer.pad_token_id,
            -100,
            labels
        )
        
        # CRITICAL FIX: Add num_items_in_batch
        inputs["num_items_in_batch"] = torch.tensor([len(batch["images"])], dtype=torch.long)
        
        return inputs
    
    except Exception as e:
        print(f"Error processing batch: {e}")
        import traceback
        traceback.print_exc()
        torch.cuda.empty_cache()
        gc.collect()
        return {
            "input_ids": torch.tensor([[0]]), 
            "attention_mask": torch.tensor([[0]]), 
            "labels": torch.tensor([[-100]]),
            "num_items_in_batch": torch.tensor([1], dtype=torch.long)
        }

# Custom Trainer
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Override to use standard PyTorch cross-entropy"""
        inputs.pop("num_items_in_batch", None)
        
        labels = inputs.pop("labels")
        
        outputs = model(**inputs)
        logits = outputs.logits
        
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        shift_labels = shift_labels.view(-1)
        
        loss = loss_fct(shift_logits, shift_labels)
        
        return (loss, outputs) if return_outputs else loss

training_args = TrainingArguments(
    output_dir=output_model,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-6,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    logging_steps=10,
    fp16=True,
    remove_unused_columns=False,
    dataloader_pin_memory=False,
    torch_compile=False,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    optim="paged_adamw_8bit",
    max_grad_norm=0.3,
    dataloader_num_workers=0,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
)

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=val_data,
    data_collator=collate_fn,
)

try:
    print("Starting training...")
    print(f"GPU Memory: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
    trainer.train()
    print("Training completed successfully!")
except Exception as e:
    print(f"Training error: {e}")
    import traceback
    traceback.print_exc()
    print("Attempting to save the model before exiting...")

try:
    print("Saving model...")
    model.save_pretrained(output_model)
    processor.save_pretrained(output_model)
    
    print("Uploading model to Hugging Face...")
    model.push_to_hub(repo_id)
    processor.push_to_hub(repo_id)
    
    print(f"Model uploaded: https://huggingface.co/{repo_id} (Private)")
    print("Fine-tuning completed!")
except Exception as e:
    print(f"Error saving or uploading model: {e}")