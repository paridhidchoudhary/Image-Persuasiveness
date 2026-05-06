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
import json
import numpy as np
from PIL import Image
from collections import defaultdict
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig, TrainingArguments, Trainer
from huggingface_hub import login
from qwen_vl_utils import process_vision_info

torch._dynamo.config.suppress_errors = True

SEED = 48
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

login(token="hf_xxxxxxxxxxxxxx")

data_root = "home/debajyoti/paridhi_mtp/product_images_real"
dataset_image = os.path.join(data_root, "dataset_image_new")
dataset_response = os.path.join(data_root, "final_data")
STAGE1_MODEL = "./vlm_stage1_lm"       # output of Stage 1
output_model = "./vlm_stage2_ranking"
TEST_IDS_FILE = "test_ids.json"        # saved by Stage 1
MAX_IMAGES = 4

# ── Load Stage 1 model — LM weights are now quality-aware ─────
print(f"Loading Stage 1 model from {STAGE1_MODEL}...")
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = AutoProcessor.from_pretrained(STAGE1_MODEL, use_fast=False)

model = AutoModelForImageTextToText.from_pretrained(
    STAGE1_MODEL,
    torch_dtype=torch.float16,
    device_map="auto",
)

if hasattr(model.config, 'use_fused_cross_entropy'):
    model.config.use_fused_cross_entropy = False
model.config.use_cache = False

# ── Freeze entire LM — only score head trains ─────────────────
print("Freezing LM parameters...")
for name, param in model.named_parameters():
    param.requires_grad = False

frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
print(f"Frozen parameters: {frozen:,}")

# ── Score head — float32, trains from scratch ─────────────────
hidden_size = model.config.hidden_size
model.score_head = nn.Sequential(
    nn.Linear(hidden_size, hidden_size // 2),
    nn.LayerNorm(hidden_size // 2),
    nn.GELU(),
    nn.Dropout(0.1),
    nn.Linear(hidden_size // 2, 1),
    nn.Sigmoid()
).to("cuda:0")  

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

model.score_head.apply(init_weights)
model.score_head = model.score_head.to("cuda:0")
trainable = sum(p.numel() for p in model.score_head.parameters())
print(f"Trainable parameters (score head only): {trainable:,}")

torch.cuda.empty_cache()
gc.collect()

# ── Dataset — load using saved test IDs ───────────────────────
from simple_data_preprocess import extract_text_data

print("Loading dataset...")
all_data = []

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
        extracted_info, ranking = extract_text_data(response_path)
        gt_scores = [img_info.get("score", 0) for img_info in extracted_info]
        if len(gt_scores) == 0 or all(s == 0 for s in gt_scores):
            continue
        all_data.append({
            "images": images,
            "scores": gt_scores,
            "category": category,
            "group": group
        })

print(f"Total loaded: {len(all_data)} groups")

# Load test IDs from Stage 1 to use exact same split
with open(TEST_IDS_FILE) as f:
    test_ids = json.load(f)
test_set = {(d["category"], d["group"]) for d in test_ids}

non_test = [d for d in all_data if (d["category"], d["group"]) not in test_set]
test_data = [d for d in all_data if (d["category"], d["group"]) in test_set]

# Split non-test into train/val
def stratified_split(data, test_ratio=0.10, min_test=1, seed=SEED):
    rng = random.Random(seed)
    by_cat = defaultdict(list)
    for item in data:
        by_cat[item['category']].append(item)
    train, val = [], []
    for cat, items in by_cat.items():
        items = items.copy()
        rng.shuffle(items)
        n_val = max(min_test, math.ceil(len(items) * test_ratio))
        n_val = min(n_val, len(items) - 1)
        val.extend(items[:n_val])
        train.extend(items[n_val:])
    return train, val

train_data, val_data = stratified_split(non_test, test_ratio=0.10)
print(f"Split: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")

# ── Losses ────────────────────────────────────────────────────

def listnet_loss(pred_scores, gt_scores):
    """ListNet — directly optimizes Kendall/Spearman."""
    P_pred = torch.softmax(pred_scores, dim=0)
    P_gt = torch.softmax(gt_scores * 5.0, dim=0)  # sharpen GT
    return -torch.sum(P_gt * torch.log(P_pred + 1e-10))


def pairwise_margin_loss(pred_scores, gt_scores, margin=0.2):
    """Pairwise with margin, skip near-ties in GT."""
    losses = []
    n = len(pred_scores)
    for i in range(n):
        for j in range(i + 1, n):
            gt_diff = gt_scores[i] - gt_scores[j]
            if abs(gt_diff.item()) < 0.05:
                continue
            if gt_diff > 0:
                loss = torch.clamp(margin - (pred_scores[i] - pred_scores[j]), min=0)
            else:
                loss = torch.clamp(margin - (pred_scores[j] - pred_scores[i]), min=0)
            losses.append(loss)
    if not losses:
        return torch.tensor(0.0, device=pred_scores.device, requires_grad=True)
    return torch.mean(torch.stack(losses))


def approx_ndcg_loss(pred_scores, gt_scores):
    """Approximate NDCG loss."""
    n = len(pred_scores)
    if n < 2:
        return torch.tensor(0.0, device=pred_scores.device, requires_grad=True)
    gt_sorted_idx = torch.argsort(gt_scores, descending=True)
    ideal_gains = gt_scores[gt_sorted_idx]
    ideal_discounts = torch.log2(
        torch.arange(2, n + 2, dtype=torch.float, device=pred_scores.device)
    )
    idcg = (ideal_gains / ideal_discounts).sum()
    if idcg == 0:
        return torch.tensor(0.0, device=pred_scores.device, requires_grad=True)
    scores_matrix = pred_scores.unsqueeze(1) - pred_scores.unsqueeze(0)
    P = torch.sigmoid(scores_matrix * 10.0)
    rank_positions = P.sum(dim=1)
    discounts = torch.log2(rank_positions + 2.0)
    gains = gt_scores / discounts
    dcg = gains.sum()
    return 1.0 - (dcg / idcg)


# ── Collate ───────────────────────────────────────────────────

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

        prompt_text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
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

        inputs["num_items_in_batch"] = torch.tensor(
            [len(batch["images"])], dtype=torch.long
        )

        # Normalize GT scores to [0,1]
        raw_scores = batch.get("scores", [])
        gt_scores = torch.tensor(raw_scores, dtype=torch.float)
        if gt_scores.max() > gt_scores.min():
            gt_scores = (gt_scores - gt_scores.min()) / (gt_scores.max() - gt_scores.min())
        else:
            gt_scores = torch.full_like(gt_scores, 0.5)

        inputs["gt_scores"] = gt_scores
        return inputs

    except Exception as e:
        print(f"Collate error: {e}")
        torch.cuda.empty_cache()
        gc.collect()
        return {
            "input_ids": torch.tensor([[0]]),
            "attention_mask": torch.tensor([[0]]),
            "num_items_in_batch": torch.tensor([1], dtype=torch.long),
            "gt_scores": torch.tensor([0.5], dtype=torch.float),
        }


# ── Trainer ───────────────────────────────────────────────────

class Stage2Trainer(Trainer):

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """Bypass Unsloth RL prediction_step."""
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            loss = self.compute_loss(model, inputs)
        return (loss.detach(), None, None)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        inputs.pop("num_items_in_batch", None)
        gt_scores = inputs.pop("gt_scores", None)

        if gt_scores is None:
            return torch.tensor(0.0, device=next(model.parameters()).device, requires_grad=True)

        # Skip dummy batches
        if gt_scores.shape[0] == 1 and gt_scores[0].item() == 0.5:
            return torch.tensor(0.0, device=next(model.parameters()).device, requires_grad=True)

        try:
            outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]
        except Exception as e:
            print(f"Forward error: {e}")
            return torch.tensor(0.0, device=next(model.parameters()).device, requires_grad=True)

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
                region = hidden_states[0, start:end, :]
                image_embeddings.append(region.mean(dim=0))
                i = end
            else:
                i += 1

        if len(image_embeddings) == 0:
            return torch.tensor(0.0, device=next(model.parameters()).device, requires_grad=True)

        # Handle partial extraction
        if len(image_embeddings) != num_images:
            print(f"Partial: {len(image_embeddings)}/{num_images}")
            gt_scores = gt_scores[:len(image_embeddings)]

        image_embeds = torch.stack(image_embeddings)

        # float32 for score head — hidden states are float16
        image_embeds_f32 = image_embeds.float()
        pred_scores = model.score_head(image_embeds_f32).squeeze(-1)
        gt_scores_dev = gt_scores.to(pred_scores.device)

        # Pure ranking loss
        listnet = listnet_loss(pred_scores, gt_scores_dev)
        pairwise = pairwise_margin_loss(pred_scores, gt_scores_dev, margin=0.2)
        ndcg = approx_ndcg_loss(pred_scores, gt_scores_dev)
        loss = 0.5 * listnet + 0.2 * pairwise + 0.3 * ndcg

        if self.state.global_step % 50 == 0:
            pred_order = torch.argsort(pred_scores, descending=True).cpu().numpy()
            gt_order = torch.argsort(gt_scores_dev, descending=True).cpu().numpy()
            print(f"\n=== Step {self.state.global_step} ===")
            print(f"ListNet={listnet.item():.4f} | Pair={pairwise.item():.4f} | NDCG={ndcg.item():.4f} | Total={loss.item():.4f}")
            print(f"Pred: {pred_scores.detach().cpu().numpy().round(3)}")
            print(f"GT:   {gt_scores_dev.cpu().numpy().round(3)}")
            print(f"Pred order: {pred_order} | GT order: {gt_order} | Match: {np.array_equal(pred_order, gt_order)}")

        return (loss, outputs) if return_outputs else loss


# ── Training args — higher LR since only score head trains ────
training_args = TrainingArguments(
    output_dir=output_model,
    overwrite_output_dir=True,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=20,        # more epochs — score head is small, converges fast
    learning_rate=1e-4,         # high LR — only score head (few params) trains
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    eval_strategy="steps",
    eval_steps=50,
    save_strategy="steps",
    save_steps=50,
    logging_steps=10,
    load_best_model_at_end=False,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    save_total_limit=3,
    fp16=False,                 # score head is float32 — avoid fp16 issues
    remove_unused_columns=False,
    dataloader_pin_memory=False,
    torch_compile=False,
    gradient_checkpointing=False,  # LM frozen — no need for checkpointing
    optim="adamw_torch",           # standard adamw — score head is small
    max_grad_norm=1.0,
    dataloader_num_workers=0,
)

trainer = Stage2Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=val_data,
    data_collator=collate_fn,
)

print("\nStarting Stage 2 — Score head training (LM frozen)...")
print(f"GPU Memory: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

# Verify score head parameters require grad — BEFORE training
for name, param in model.score_head.named_parameters():
    param.requires_grad = True
    print(f"Score head {name}: requires_grad={param.requires_grad}, device={param.device}")

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total trainable params: {trainable:,}")

try:
    trainer.train()
except Exception as e:
    print(f"Training error: {e}")
    import traceback
    traceback.print_exc()
    raise

print("Saving Stage 2 model...")
os.makedirs(output_model, exist_ok=True)
score_head_path = os.path.join(output_model, "score_head.pt")
torch.save(model.score_head.state_dict(), score_head_path)
print(f"✓ Score head saved to {score_head_path}")

processor.save_pretrained(output_model)

config = {"stage1_model": STAGE1_MODEL, "score_head": score_head_path}
with open(os.path.join(output_model, "stage2_config.json"), "w") as f:
    json.dump(config, f, indent=2)

print(f"Stage 2 done!")
print(f"Next: run eval_v2.py (update FINETUNED_MODEL_NAME='{output_model}')")