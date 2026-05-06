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
import random 
import numpy as np
import torch.nn as nn

torch._dynamo.config.suppress_errors = True

SEED = 50
random.seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

login(token="hf_xxxxxxxxxxxxxxxxx")

data_root = "home/debajyoti/paridhi_mtp/product_images_real"
dataset_image = os.path.join(data_root, "dataset_image_new")
dataset_response = os.path.join(data_root, "final_data")
output_model = "./vlm_finetuned_full_context_48_2"
MAX_IMAGES = 4

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

model, tokenizer = FastVisionModel.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    use_gradient_checkpointing="unsloth", 
    device_map="auto",
    max_seq_length=2048,
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

# Load dataset
data = []
from simple_data_preprocess import extract_text_data

print("Loading dataset...")

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
                
                if len(images) > MAX_IMAGES:
                    continue
                    
                response_path = os.path.join(dataset_response, category, group, "user_output.txt")
                if os.path.exists(response_path):
                    with open(response_path, 'r') as f:
                        full_response_text = f.read()
                    
                    if len(full_response_text) > 2000:
                        full_response_text = full_response_text[:2000]
                    
                    extracted_info, ranking = extract_text_data(response_path)
                    
                    data.append({
                        "images": images,
                        "full_response_text": full_response_text,
                        "ranking": ranking,
                        "category": category,
                        "group": group
                    })

print(f"Dataset loaded with {len(data)} groups")
train_data, test_data = train_test_split(data, test_size=0.05, random_state=48)
train_data, val_data = train_test_split(train_data, test_size=0.05, random_state=48)
print(f"Training: {len(train_data)}, Validation: {len(val_data)}")

def resize_if_needed(img, max_size=256):
    if img.width > max_size or img.height > max_size:
        scaling_factor = max_size / float(max(img.width, img.height))
        new_width = int(img.width * scaling_factor)
        new_height = int(img.height * scaling_factor)
        return img.resize((new_width, new_height), Image.LANCZOS)
    return img

def collate_fn(batch):
    batch = batch[0]
    try:
        sorted_images = []
        for img_path in batch["images"]:
            img = resize_if_needed(Image.open(img_path).convert("RGB"))
            sorted_images.append({"type": "image", "image": img})
        
        category = batch["category"]
        group = batch["group"]
        
        messages = [
            {
                "role": "user",
                "content": sorted_images + [
                    {
                        "type": "text", 
                        "text": f"You are evaluating images in the '{group}' group under the '{category}' product category. "
                                f"Rank the images, based on their appeal for selling '{category}' product. "
                                f"Provide description, and *persuasion score (1-100)* for each image and explain the ranking."
                    }
                ],
            }
        ]
        
        full_response_text = batch["full_response_text"]
        assistant_message = {"role": "assistant", "content": full_response_text}
        
        prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        full_text = processor.apply_chat_template(messages + [assistant_message], tokenize=False, add_generation_prompt=False)
        
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = processor(
            text=prompt_text,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            truncation=True,
            max_length=2048,
            return_tensors="pt",
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
            labels = torch.cat([labels, padding], dim=-1)
        elif labels_len > input_len:
            labels = labels[:, :input_len]
            
        inputs["labels"] = torch.where(
            inputs["input_ids"] == processor.tokenizer.pad_token_id,
            -100,
            labels
        )
        
        inputs["num_items_in_batch"] = torch.tensor([len(batch["images"])], dtype=torch.long)
        
        return inputs
    except Exception as e:
        print(f"Error in collate: {e}")
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


# Custom Trainer to override compute_loss and use standard loss
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Override to use standard PyTorch cross-entropy instead of unsloth's fused loss
        """
        # Remove num_items_in_batch if present
        inputs.pop("num_items_in_batch", None)
        
        labels = inputs.pop("labels")
        
        # Forward pass
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Shift for causal LM
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Flatten and compute standard cross-entropy
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
)

# Use CustomTrainer instead of Trainer
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=val_data,
    data_collator=collate_fn,
)

print("Starting training...")
print(f"GPU Memory: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

try:
    trainer.train()
except Exception as e:
    print(f"Training error: {e}")
    import traceback
    traceback.print_exc()
    torch.cuda.empty_cache()
    gc.collect()
    raise

print("Saving model...")
model.save_pretrained(output_model)
processor.save_pretrained(output_model)
print("Done!")