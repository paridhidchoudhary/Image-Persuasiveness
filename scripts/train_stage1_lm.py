import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn as nn
import gc
import random
import math
import json
import numpy as np
from PIL import Image
from collections import defaultdict
from transformers import AutoProcessor, BitsAndBytesConfig, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType
from huggingface_hub import login
from qwen_vl_utils import process_vision_info
from unsloth import FastVisionModel

torch._dynamo.config.suppress_errors = True

SEED = 48
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

login(token="hf_xxxxxxxxxxxxxxxxx")

data_root = "home/debajyoti/paridhi_mtp/product_images_real"
dataset_image = os.path.join(data_root, "dataset_image_new")
dataset_response = os.path.join(data_root, "final_data")
output_model = "./vlm_stage1_lm"
MAX_IMAGES = 4

# ── Model ─────────────────────────────────────────────────────
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

# LoRA r=16 — enough capacity to learn output format
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

torch.cuda.empty_cache()
gc.collect()

# ── Dataset ───────────────────────────────────────────────────
from simple_data_preprocess import extract_text_data

print("Loading dataset...")
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
        response_path = os.path.join(dataset_response, category, group, "user_output.txt")
        if not os.path.exists(response_path):
            continue
        with open(response_path, 'r') as f:
            full_response_text = f.read()
        if len(full_response_text) > 2000:
            full_response_text = full_response_text[:2000]

        extracted_info, ranking = extract_text_data(response_path)
        gt_scores = [img_info.get("score", 0) for img_info in extracted_info]

        if len(gt_scores) == 0 or all(s == 0 for s in gt_scores):
            continue

        data.append({
            "images": images,
            "full_response_text": full_response_text,
            "ranking": ranking,
            "scores": gt_scores,
            "category": category,
            "group": group
        })

print(f"Total loaded: {len(data)} groups")

# ── Stratified split ──────────────────────────────────────────
def stratified_split(data, test_ratio=0.10, min_test=2, seed=SEED):
    rng = random.Random(seed)
    by_cat = defaultdict(list)
    for item in data:
        by_cat[item['category']].append(item)
    train, test = [], []
    for cat, items in by_cat.items():
        items = items.copy()
        rng.shuffle(items)
        n_test = max(min_test, math.ceil(len(items) * test_ratio))
        n_test = min(n_test, len(items) - 1)
        test.extend(items[:n_test])
        train.extend(items[n_test:])
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test

train_data, test_data = stratified_split(data, test_ratio=0.10, min_test=2)
train_data, val_data = stratified_split(train_data, test_ratio=0.10, min_test=1)

print(f"Split: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")

# Save test IDs — used by both Stage 2 and eval
test_ids = [{"category": d["category"], "group": d["group"]} for d in test_data]
with open("test_ids.json", "w") as f:
    json.dump(test_ids, f, indent=2)
print(f"✓ Test IDs saved to test_ids.json")

# Verify coverage
test_cats = defaultdict(int)
for item in test_data:
    test_cats[item['category']] += 1
print("Test categories:", dict(sorted(test_cats.items())))
missing = set(d['category'] for d in data) - set(test_cats.keys())
if missing:
    print(f"⚠️  Missing from test: {missing}")
else:
    print("✓ All categories in test set")

# ── Collate — LM only, no score head ─────────────────────────
def resize_if_needed(img, max_size=256):
    if img.width > max_size or img.height > max_size:
        factor = max_size / float(max(img.width, img.height))
        return img.resize((int(img.width * factor), int(img.height * factor)), Image.LANCZOS)
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

        messages = [{
            "role": "user",
            "content": sorted_images + [{
                "type": "text",
                "text": f"You are evaluating images in the '{group}' group under the '{category}' product category. "
                        f"Rank the images, based on their appeal for selling '{category}' product. "
                        f"Provide description, and *persuasion score (1-100)* for each image and explain the ranking."
            }]
        }]

        assistant_message = {
            "role": "assistant",
            "content": batch["full_response_text"]
        }

        prompt_text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        full_text = processor.apply_chat_template(
            messages + [assistant_message], tokenize=False, add_generation_prompt=False
        )

        image_inputs, video_inputs = process_vision_info(messages)

        # Encode prompt only (for masking)
        prompt_inputs = processor(
            text=prompt_text,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            truncation=True,
            max_length=2048,
            return_tensors="pt",
        )

        # Encode full text (prompt + response)
        full_inputs = processor(
            text=full_text,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            truncation=True,
            max_length=2048,
            return_tensors="pt",
        )

        # Build labels — mask prompt tokens with -100, only train on response
        prompt_len = prompt_inputs["input_ids"].shape[1]
        full_len = full_inputs["input_ids"].shape[1]

        labels = full_inputs["input_ids"].clone()
        # Mask everything up to end of prompt
        labels[:, :prompt_len] = -100
        # Mask padding
        labels[full_inputs["attention_mask"] == 0] = -100

        full_inputs["labels"] = labels
        full_inputs["num_items_in_batch"] = torch.tensor(
            [len(batch["images"])], dtype=torch.long
        )

        return full_inputs

    except Exception as e:
        print(f"Collate error: {e}")
        import traceback
        traceback.print_exc()
        torch.cuda.empty_cache()
        gc.collect()
        return {
            "input_ids": torch.tensor([[0]]),
            "attention_mask": torch.tensor([[0]]),
            "labels": torch.tensor([[-100]]),
            "num_items_in_batch": torch.tensor([1], dtype=torch.long),
        }


# ── Trainer — pure LM loss ────────────────────────────────────
class LMTrainer(Trainer):

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """Bypass Unsloth RL prediction_step."""
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            loss = self.compute_loss(model, inputs)
        return (loss.detach(), None, None)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        inputs.pop("num_items_in_batch", None)
        labels = inputs.pop("labels", None)

        if labels is None:
            return torch.tensor(0.0, device=model.device, requires_grad=True)

        try:
            outputs = model(**inputs)
            logits = outputs.logits
        except Exception as e:
            print(f"Forward error: {e}")
            return torch.tensor(0.0, device=model.device, requires_grad=True)

        # Standard cross-entropy on response tokens only (prompt masked to -100)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )

        if self.state.global_step % 50 == 0:
            print(f"\n=== Step {self.state.global_step} | LM Loss: {loss.item():.4f} ===")

        return (loss, outputs) if return_outputs else loss


# ── Training args ─────────────────────────────────────────────
training_args = TrainingArguments(
    output_dir=output_model,
    overwrite_output_dir=True,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=10,
    learning_rate=5e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    eval_strategy="steps",
    eval_steps=50,
    save_strategy="steps",
    save_steps=50,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    save_total_limit=3,
    fp16=True,
    remove_unused_columns=False,
    dataloader_pin_memory=False,
    torch_compile=False,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    optim="paged_adamw_8bit",
    max_grad_norm=1.0,
    dataloader_num_workers=0,
)

trainer = LMTrainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=val_data,
    data_collator=collate_fn,
)

print("\nStarting Stage 1 — LM fine-tuning (pure text generation)...")
print(f"GPU Memory: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

try:
    trainer.train()
except Exception as e:
    print(f"Training error: {e}")
    import traceback
    traceback.print_exc()
    raise

print("Saving Stage 1 model...")
model.save_pretrained(output_model)
processor.save_pretrained(output_model)
print(f"Stage 1 done! Model saved to {output_model}")
