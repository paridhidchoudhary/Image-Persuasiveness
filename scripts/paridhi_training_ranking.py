import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


import torch
import torch.nn as nn
import torch.nn.functional as F
import gc
import random
import math
import numpy as np
from PIL import Image
from collections import defaultdict
from transformers import AutoProcessor, BitsAndBytesConfig, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType
from huggingface_hub import login
from qwen_vl_utils import process_vision_info
from unsloth import FastVisionModel

torch._dynamo.config.suppress_errors = True

# ============================================================
# Seeds
# ============================================================
SEED = 48
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

login(token="hf_xxxxxxxxxxxxxxxxxxx")

# ============================================================
# Paths
# ============================================================
data_root = "home/debajyoti/paridhi_mtp/product_images_real"
dataset_image = os.path.join(data_root, "dataset_image_new")
dataset_response = os.path.join(data_root, "final_data")
output_model = "./vlm_finetuned_ranking_v2"
MAX_IMAGES = 4

# ============================================================
# Model — LoRA r=16, more target modules
# ============================================================
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

# LoRA r=16, targeting q/k/v/o for richer vision representations
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=16,           # was 4 — 4x more capacity
    lora_alpha=32,  # was 16
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # was only q/v
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

hidden_size = model.config.hidden_size

# Score head — same architecture, kept float32
model.score_head = nn.Sequential(
    nn.Linear(hidden_size, hidden_size // 2),
    nn.LayerNorm(hidden_size // 2),
    nn.GELU(),
    nn.Dropout(0.1),
    nn.Linear(hidden_size // 2, 1),
    nn.Sigmoid()
).to(model.device)

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

model.score_head.apply(init_weights)
torch.cuda.empty_cache()
gc.collect()

# ============================================================
# Dataset loading with stratified split
# ============================================================
from simple_data_preprocess import extract_text_data

print("Loading dataset...")
data = []

for category in os.listdir(dataset_image):
    category_path = os.path.join(dataset_image, category)
    if not os.path.isdir(category_path):
        continue

    for group in os.listdir(category_path):
        group_path = os.path.join(dataset_image, category, group)
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

# Category distribution
cat_counts = defaultdict(int)
for item in data:
    cat_counts[item['category']] += 1
print("Category distribution:")
for cat in sorted(cat_counts.keys()):
    print(f"  {cat}: {cat_counts[cat]}")

# ============================================================
# FIX: Stratified split guaranteeing all categories in test
# ============================================================
def stratified_split(data, test_ratio=0.10, min_test=2, seed=SEED):
    """
    Guarantee at least min_test groups per category in test set.
    Uses test_ratio=10% (was 5%) to get more test samples.
    """
    rng = random.Random(seed)
    by_cat = defaultdict(list)
    for item in data:
        by_cat[item['category']].append(item)

    train, test = [], []
    for cat, items in by_cat.items():
        items = items.copy()
        rng.shuffle(items)
        n_test = max(min_test, math.ceil(len(items) * test_ratio))
        n_test = min(n_test, len(items) - 1)  # always keep at least 1 in train
        test.extend(items[:n_test])
        train.extend(items[n_test:])

    rng.shuffle(train)
    rng.shuffle(test)
    return train, test

train_data, test_data = stratified_split(data, test_ratio=0.10, min_test=2)

# Further split train into train/val
train_data, val_data = stratified_split(train_data, test_ratio=0.10, min_test=1)

print(f"\nSplit: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")

# Verify all categories in test
test_cats = defaultdict(int)
for item in test_data:
    test_cats[item['category']] += 1
print("Test categories:")
for cat in sorted(test_cats.keys()):
    print(f"  {cat}: {test_cats[cat]}")

missing = set(cat_counts.keys()) - set(test_cats.keys())
if missing:
    print(f"⚠️  Still missing from test: {missing}")
else:
    print("✓ All categories represented in test set")

# Save test group identifiers so eval uses exact same split
import json
test_ids = [{"category": d["category"], "group": d["group"]} for d in test_data]
with open(os.path.join(output_model + "_test_ids.json"), "w") as f:
    json.dump(test_ids, f, indent=2)
print(f"Test IDs saved to {output_model}_test_ids.json")

# ============================================================
# Losses — pure ranking, no LM loss
# ============================================================

def listnet_loss(pred_scores, gt_scores):
    """
    ListNet top-1 probability loss.
    Directly optimizes Kendall/Spearman by matching full ranking distributions.
    """
    P_pred = torch.softmax(pred_scores, dim=0)
    P_gt = torch.softmax(gt_scores * 5.0, dim=0)  # temperature=5 sharpens GT distribution
    loss = -torch.sum(P_gt * torch.log(P_pred + 1e-10))
    return loss


def pairwise_margin_loss(pred_scores, gt_scores, margin=0.2):
    """
    Pairwise ranking with margin=0.2 (was 0.1 — too small for [0,1] range).
    Skips pairs where GT scores are too close (within 0.05) to avoid noise.
    """
    losses = []
    n = len(pred_scores)
    for i in range(n):
        for j in range(i + 1, n):
            gt_diff = gt_scores[i] - gt_scores[j]
            if abs(gt_diff.item()) < 0.05:
                continue  # skip near-ties in GT — noisy signal
            if gt_diff > 0:
                loss = torch.clamp(margin - (pred_scores[i] - pred_scores[j]), min=0)
            else:
                loss = torch.clamp(margin - (pred_scores[j] - pred_scores[i]), min=0)
            losses.append(loss)

    if len(losses) == 0:
        return torch.tensor(0.0, device=pred_scores.device, requires_grad=True)
    return torch.mean(torch.stack(losses))


def approx_ndcg_loss(pred_scores, gt_scores):
    """
    Approximated NDCG loss — directly optimizes ranking quality.
    Uses a smooth approximation via softmax.
    """
    n = len(pred_scores)
    if n < 2:
        return torch.tensor(0.0, device=pred_scores.device, requires_grad=True)

    # Ideal DCG (from GT)
    gt_sorted_idx = torch.argsort(gt_scores, descending=True)
    ideal_gains = gt_scores[gt_sorted_idx]
    ideal_discounts = torch.log2(torch.arange(2, n + 2, dtype=torch.float, device=pred_scores.device))
    idcg = (ideal_gains / ideal_discounts).sum()

    if idcg == 0:
        return torch.tensor(0.0, device=pred_scores.device, requires_grad=True)

    # Soft permutation matrix via Sinkhorn (approximation via softmax)
    scores_matrix = pred_scores.unsqueeze(1) - pred_scores.unsqueeze(0)
    P = torch.sigmoid(scores_matrix * 10.0)  # soft ordering
    rank_positions = P.sum(dim=1)  # soft rank for each item

    discounts = torch.log2(rank_positions + 2.0)
    gains = gt_scores / discounts
    dcg = gains.sum()

    ndcg = dcg / idcg
    return 1.0 - ndcg  # loss = 1 - NDCG


# ============================================================
# Collate — same image processing, score normalization
# ============================================================

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

        prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
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

        inputs["num_items_in_batch"] = torch.tensor([len(batch["images"])], dtype=torch.long)

        # Normalize GT scores to [0,1]
        raw_scores = batch.get("scores", [])
        if len(raw_scores) == 0:
            gt_scores = torch.zeros(len(batch["images"]), dtype=torch.float)
        else:
            gt_scores = torch.tensor(raw_scores, dtype=torch.float)
            if gt_scores.max() > gt_scores.min():
                gt_scores = (gt_scores - gt_scores.min()) / (gt_scores.max() - gt_scores.min())
            else:
                gt_scores = torch.full_like(gt_scores, 0.5)

        inputs["gt_scores"] = gt_scores
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
            "num_items_in_batch": torch.tensor([1], dtype=torch.long),
            "gt_scores": torch.tensor([0.5], dtype=torch.float)
        }


# ============================================================
# Custom Trainer — pure ranking loss, no LM loss
# ============================================================

class RankingTrainer(Trainer):
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """
        Override to bypass Unsloth's RL prediction_step which expects a 'prompt' key.
        We just compute loss on the eval batch using our own compute_loss.
        """
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            loss = self.compute_loss(model, inputs)
        loss = loss.detach()
        return (loss, None, None)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        inputs.pop("num_items_in_batch", None)
        gt_scores = inputs.pop("gt_scores", None)

        if gt_scores is None:
            return torch.tensor(0.0, device=model.device, requires_grad=True)

        # Skip dummy batches
        if gt_scores.shape[0] == 1 and gt_scores[0] == 0.5:
            return torch.tensor(0.0, device=model.device, requires_grad=True)

        # Forward pass — only need hidden states, not logits
        try:
            outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]
        except Exception as e:
            print(f"Forward pass error: {e}")
            return torch.tensor(0.0, device=model.device, requires_grad=True)

        # Extract image embeddings from vision token regions
        num_images = gt_scores.shape[0]
        vision_start_id = 151652
        vision_end_id = 151653

        input_ids_list = inputs['input_ids'][0].tolist()
        image_embeddings = []

        i = 0
        while i < len(input_ids_list) and len(image_embeddings) < num_images:
            if input_ids_list[i] == vision_start_id:
                start = i
                j = i + 1
                while j < len(input_ids_list) and input_ids_list[j] != vision_end_id:
                    j += 1
                end = j + 1
                image_region = hidden_states[0, start:end, :]
                pooled = image_region.mean(dim=0)
                image_embeddings.append(pooled)
                i = end
            else:
                i += 1

        if len(image_embeddings) == 0:
            print(f"Warning: no embeddings found, skipping batch")
            return torch.tensor(0.0, device=model.device, requires_grad=True)
        
        if len(image_embeddings) != num_images:
            print(f"Warning: extracted {len(image_embeddings)}/{num_images}, using partial")
            gt_scores = gt_scores[:len(image_embeddings)]  # truncate BEFORE anything else
        
        image_embeds = torch.stack(image_embeddings)
        image_embeds_f32 = image_embeds.float()
        pred_scores = model.score_head(image_embeds_f32).squeeze(-1)
        gt_scores_dev = gt_scores.to(pred_scores.device)  # defined AFTER truncation

        
        # ============================================================
        # Pure ranking loss — no LM loss at all
        # Weights: ListNet (ranking distribution) + pairwise + NDCG
        # ============================================================
        listnet = listnet_loss(pred_scores, gt_scores_dev)
        pairwise = pairwise_margin_loss(pred_scores, gt_scores_dev, margin=0.2)
        ndcg = approx_ndcg_loss(pred_scores, gt_scores_dev)

        # Weight toward ListNet (best for Kendall/Spearman) + NDCG for top accuracy
        loss = 0.5 * listnet + 0.2 * pairwise + 0.3 * ndcg

        if self.state.global_step % 50 == 0:
            pred_order = torch.argsort(pred_scores, descending=True).cpu().numpy()
            gt_order = torch.argsort(gt_scores_dev, descending=True).cpu().numpy()
            print(f"\n=== Step {self.state.global_step} ===")
            print(f"ListNet: {listnet.item():.4f} | Pairwise: {pairwise.item():.4f} | NDCG: {ndcg.item():.4f}")
            print(f"Total: {loss.item():.4f}")
            print(f"Pred: {pred_scores.detach().cpu().numpy().round(3)}")
            print(f"GT:   {gt_scores_dev.cpu().numpy().round(3)}")
            print(f"Pred order: {pred_order} | GT order: {gt_order} | Match: {np.array_equal(pred_order, gt_order)}")

        return (loss, outputs) if return_outputs else loss


# ============================================================
# Training args — multi-GPU via device_map="auto"
# ============================================================
training_args = TrainingArguments(
    output_dir=output_model,
    overwrite_output_dir=True,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,   # effective batch = 8 (2 GPUs × 1 × 4 accum)
    num_train_epochs=15,             # more epochs since we dropped LM loss complexity
    learning_rate=2e-4,              # higher LR — score head trains from scratch
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

trainer = RankingTrainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=val_data,
    data_collator=collate_fn,
)

print("\nStarting training...")
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

score_head_path = os.path.join(output_model, "score_head.pt")
torch.save(model.score_head.state_dict(), score_head_path)
print(f"Score head saved to {score_head_path}")
print("Done!")