"""
Step 2a: Attention-weighted pooling instead of mean+max pooling.

Key differences from option 1:
- AttentionPooling module learns WHICH vision tokens matter most for persuasiveness
  instead of treating all tokens equally (mean) or taking the maximum (max)
- Score head input is hidden_size (not hidden_size*2) since no concatenation
- Both AttentionPooling and score head are trained; LoRA backbone also unfrozen
- Loads phase 1 score head as warm start for score head weights
  (note: score head input dim changed so only partial warm start is possible)

Why this might help:
  Mean pooling averages everything equally — background tokens dilute signal.
  Max pooling takes extreme values — noisy for ranking.
  Attention pooling LEARNS to focus on the tokens that discriminate image quality,
  e.g., product clarity, background clutter, lighting highlights.
"""

import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import unsloth
from unsloth import FastVisionModel
import torch
import gc
import re
from PIL import Image
from sklearn.model_selection import train_test_split
from transformers import AutoProcessor, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType
from huggingface_hub import login
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

login(token="hf_xxxxxxxxxxx")

# ── Paths ──────────────────────────────────────────────────────────────────────
# PHASE1_CHECKPOINT = "./vlm_finetuned_listwise"       # for score head warm start
OUTPUT_MODEL      = "./vlm_finetuned_attn_pooling"   # step 2a output

data_root         = "home/debajyoti/paridhi_mtp/product_images_real"
dataset_image     = os.path.join(data_root, "dataset_image_new")
dataset_response  = os.path.join(data_root, "final_data")
MAX_IMAGES        = 4

# ── Global normalization ───────────────────────────────────────────────────────
def get_global_normalization(dataset_root):
    all_scores = []
    for category in os.listdir(dataset_root):
        category_path = os.path.join(dataset_root, category)
        if not os.path.isdir(category_path):
            continue
        for group in os.listdir(category_path):
            group_path = os.path.join(category_path, group)
            gt_file = os.path.join(group_path, "user_output.txt")
            if not os.path.exists(gt_file):
                continue
            with open(gt_file, "r") as f:
                text = f.read()
            scores = re.findall(r"score\s*[:\-]?\s*(\d+)", text.lower())
            for s in scores:
                all_scores.append(float(s))
    global_min = min(all_scores)
    global_max = max(all_scores)
    print(f"Global score range: {global_min} → {global_max}")
    return global_min, global_max

GLOBAL_MIN, GLOBAL_MAX = get_global_normalization(dataset_response)

# ── Load model ─────────────────────────────────────────────────────────────────
model_name = "unsloth/Qwen2.5-VL-7B-Instruct"
processor  = AutoProcessor.from_pretrained(model_name, use_fast=False)

torch.cuda.empty_cache()
gc.collect()

model, tokenizer = FastVisionModel.from_pretrained(
    model_name,
    load_in_4bit=True,
    use_gradient_checkpointing="unsloth",
    device_map="auto",
    max_seq_length=2048,
)

if hasattr(model.config, 'use_fused_cross_entropy'):
    model.config.use_fused_cross_entropy = False
model.config.use_cache = False

for name, module in model.named_modules():
    for param_name, param in module.named_parameters(recurse=False):
        if param.dtype == torch.float32:
            param.data = param.data.to(torch.bfloat16)
    for buf_name, buf in module.named_buffers(recurse=False):
        if buf.dtype == torch.float32:
            module._buffers[buf_name] = buf.to(torch.bfloat16)

# ── LoRA — same as option 1 ────────────────────────────────────────────────────
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
)
model = get_peft_model(model, peft_config)

model_dtype = next(
    (p.dtype for p in model.parameters() if p.dtype in (torch.float16, torch.bfloat16)),
    torch.float16
)
print(f"Model compute dtype: {model_dtype}")

hidden_size = model.config.hidden_size  # NOTE: NOT *2 — attention pooling outputs hidden_size

# ── Attention Pooling module ───────────────────────────────────────────────────
class AttentionPooling(nn.Module):
    """
    Learns a scalar attention weight for each vision token position.
    Weighted sum of token embeddings → single vector per image.
    
    Why this is better than mean+max:
    - Mean pooling: background/padding tokens dilute signal equally
    - Max pooling: picks extremes, noisy for subtle quality differences  
    - Attention: LEARNS which tokens carry ranking-relevant information
      (e.g., product clarity, foreground sharpness, background clutter)
    """
    def __init__(self, hidden_size):
        super().__init__()
        # Single linear layer outputs a scalar weight per token
        self.attention = nn.Linear(hidden_size, 1, bias=True)

    def forward(self, image_region):
        """
        image_region: [seq_len, hidden_size] — vision tokens for one image
        Returns: [hidden_size] — attention-weighted pooled embedding
        """
        # Compute raw attention scores: [seq_len, 1]
        attn_scores = self.attention(image_region)
        # Normalize to probability distribution over tokens: [seq_len, 1]
        attn_weights = torch.softmax(attn_scores, dim=0)
        # Weighted sum: [hidden_size]
        pooled = (attn_weights * image_region).sum(dim=0)
        return pooled

# ── Score head — input is hidden_size (not hidden_size*2) ─────────────────────
model.attn_pooling = AttentionPooling(hidden_size).to(model.device).to(model_dtype)

model.score_head = nn.Sequential(
    nn.Linear(hidden_size, hidden_size // 2),
    nn.LayerNorm(hidden_size // 2),
    nn.GELU(),
    nn.Dropout(0.1),
    nn.Linear(hidden_size // 2, 1),
    nn.Sigmoid()
).to(model.device).to(model_dtype)

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

model.attn_pooling.apply(init_weights)
model.score_head.apply(init_weights)
model.attn_pooling = model.attn_pooling.to(model_dtype)
model.score_head   = model.score_head.to(model_dtype)

# Note: we do NOT load phase 1 score head weights here because the input
# dimension changed (hidden_size vs hidden_size*2). Training from scratch
# for both attn_pooling and score_head, but LoRA warm-started from pretrained.
print("✓ AttentionPooling and score head initialized from scratch (dim changed)")
print(f"  Score head input dim: {hidden_size} (was {hidden_size*2} in option 1)")

# ── Freeze all, unfreeze LoRA + attn_pooling + score_head ─────────────────────
for param in model.parameters():
    param.requires_grad = False

lora_count = 0
for name, param in model.named_parameters():
    if 'lora_' in name:
        param.requires_grad = True
        lora_count += param.numel()

for param in model.attn_pooling.parameters():
    param.requires_grad = True
for param in model.score_head.parameters():
    param.requires_grad = True

attn_count  = sum(p.numel() for p in model.attn_pooling.parameters())
head_count  = sum(p.numel() for p in model.score_head.parameters())
print(f"✓ Trainable: {lora_count:,} LoRA + {attn_count:,} attn_pooling + {head_count:,} score_head")

torch.cuda.empty_cache()
gc.collect()

# ── Dataset ────────────────────────────────────────────────────────────────────
from simple_data_preprocess import extract_text_data

data = []
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
                    gt_scores = [img_info.get("score", 0) for img_info in extracted_info]
                    if len(set(gt_scores)) == 1:
                        continue
                    data.append({
                        "images": images,
                        "full_response_text": full_response_text,
                        "ranking": ranking,
                        "scores": gt_scores,
                        "category": category,
                        "group": group,
                    })

print(f"Dataset loaded with {len(data)} groups")
train_data, test_data = train_test_split(data, test_size=0.15, random_state=48)
train_data, val_data  = train_test_split(train_data, test_size=0.05, random_state=48)
print(f"Training: {len(train_data)}, Validation: {len(val_data)}, Test: {len(test_data)}")

# ── Collate (identical to option 1) ───────────────────────────────────────────
def resize_if_needed(img, max_size=256):
    if img.width > max_size or img.height > max_size:
        scale = max_size / float(max(img.width, img.height))
        return img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)
    return img

def collate_fn(batch):
    batch = batch[0]
    try:
        sorted_images = []
        for img_path in batch["images"]:
            img = resize_if_needed(Image.open(img_path).convert("RGB"))
            sorted_images.append({"type": "image", "image": img})

        category = batch["category"]
        group    = batch["group"]

        messages = [{
            "role": "user",
            "content": sorted_images + [{
                "type": "text",
                "text": (f"You are evaluating images in the '{group}' group under the "
                         f"'{category}' product category. Rank the images, based on their "
                         f"appeal for selling '{category}' product. Provide description, "
                         f"and *persuasion score (1-100)* for each image and explain the ranking.")
            }],
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
            return_tensors="pt",
        )

        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

        inputs["num_items_in_batch"] = torch.tensor([len(batch["images"])], dtype=torch.long)

        if "scores" not in batch or len(batch["scores"]) == 0:
            print(f"Warning: No scores for {category}/{group}, using dummy scores")
            gt_scores = torch.zeros(len(batch["images"]), dtype=torch.float)
        else:
            gt_scores = torch.tensor(batch["scores"], dtype=torch.float)
            gt_scores = (gt_scores - GLOBAL_MIN) / (GLOBAL_MAX - GLOBAL_MIN)

        inputs["gt_scores"] = gt_scores
        return inputs

    except Exception as e:
        print(f"Error in collate: {e}")
        import traceback
        traceback.print_exc()
        torch.cuda.empty_cache()
        gc.collect()
        return {
            "input_ids":          torch.tensor([[0]]),
            "attention_mask":     torch.tensor([[0]]),
            "num_items_in_batch": torch.tensor([1], dtype=torch.long),
            "gt_scores":          torch.tensor([0.0], dtype=torch.float),
        }

# ── Loss functions (identical) ─────────────────────────────────────────────────
def listnet_loss(pred_scores, gt_scores):
    P_pred = torch.softmax(pred_scores, dim=0)
    P_gt   = torch.softmax(gt_scores,   dim=0)
    return -torch.sum(P_gt * torch.log(P_pred + 1e-10))

def pairwise_margin_loss(pred_scores, gt_scores, margin=0.02):
    losses = []
    n = len(pred_scores)
    for i in range(n):
        for j in range(i + 1, n):
            if gt_scores[i] > gt_scores[j]:
                losses.append(torch.clamp(margin - (pred_scores[i] - pred_scores[j]), min=0))
            elif gt_scores[j] > gt_scores[i]:
                losses.append(torch.clamp(margin - (pred_scores[j] - pred_scores[i]), min=0))
    if not losses:
        return torch.tensor(0.0, device=pred_scores.device, dtype=pred_scores.dtype)
    return torch.mean(torch.stack(losses))

def mse_loss_normalized(pred_scores, gt_scores):
    return torch.mean((pred_scores - gt_scores) ** 2)

# ── Trainer ────────────────────────────────────────────────────────────────────
class AttnPoolingTrainer(Trainer):
    def __init__(self, score_head, attn_pooling, **kwargs):
        super().__init__(**kwargs)
        self.score_head   = score_head
        self.attn_pooling = attn_pooling

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            loss = self.compute_loss(model, inputs)
        return (loss.detach(), None, None)

    def create_optimizer(self):
        lora_params        = [p for n, p in self.model.named_parameters()
                              if 'lora_' in n and p.requires_grad]
        head_params        = list(self.score_head.parameters())
        attn_params        = list(self.attn_pooling.parameters())

        self.optimizer = torch.optim.AdamW(
            [
                {"params": lora_params,  "lr": self.args.learning_rate},       # 1e-5
                {"params": head_params,  "lr": self.args.learning_rate * 3},   # 3e-5
                {"params": attn_params,  "lr": self.args.learning_rate * 3},   # 3e-5
            ],
            weight_decay=self.args.weight_decay,
        )
        return self.optimizer

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        inputs.pop("num_items_in_batch", None)
        gt_scores = inputs.pop("gt_scores", None)
        inputs.pop("labels", None)

        if gt_scores is None:
            return torch.tensor(0.0, device=model.device, requires_grad=True)
        if gt_scores.shape[0] == 1 and gt_scores[0] == 0.0:
            return torch.tensor(0.0, device=model.device, requires_grad=True)

        try:
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]
        except Exception as e:
            print(f"Error in forward pass: {e}")
            return torch.tensor(0.0, device=model.device, requires_grad=True)

        num_images      = gt_scores.shape[0]
        vision_start_id = processor.tokenizer.convert_tokens_to_ids("<|vision_start|>")
        vision_end_id   = processor.tokenizer.convert_tokens_to_ids("<|vision_end|>")
        input_ids_list  = inputs['input_ids'][0].tolist()
        image_embeddings = []

        i = 0
        while i < len(input_ids_list) and len(image_embeddings) < num_images:
            if input_ids_list[i] == vision_start_id:
                start = i
                j = i + 1
                while j < len(input_ids_list) and input_ids_list[j] != vision_end_id:
                    j += 1
                end = j + 1
                image_region = hidden_states[0, start + 1:end - 1, :]  # [seq_len, hidden_size]

                # ── Attention pooling (replaces mean+max cat) ──────────────────
                image_region = image_region.to(self.attn_pooling.attention.weight.dtype)
                pooled = self.attn_pooling(image_region)  # [hidden_size]
                image_embeddings.append(pooled)
                i = end
            else:
                i += 1

        if len(image_embeddings) != num_images:
            print(f"Warning: Could not extract {num_images} image embeddings, skipping")
            return torch.tensor(0.0, device=model.device, requires_grad=True)

        image_embeds   = torch.stack(image_embeddings)  # [num_images, hidden_size]
        image_embeds   = image_embeds.to(next(self.score_head.parameters()).dtype)
        pred_scores    = self.score_head(image_embeds).squeeze(-1)
        gt_scores_norm = gt_scores.to(pred_scores.device).to(pred_scores.dtype)

        listnet   = listnet_loss(pred_scores, gt_scores_norm)
        pairwise  = pairwise_margin_loss(pred_scores, gt_scores_norm, margin=0.02)
        mse       = mse_loss_normalized(pred_scores, gt_scores_norm)
        rank_loss = 0.3 * listnet + 0.2 * pairwise + 0.5 * mse

        if self.state.global_step % 50 == 0:
            print(f"\n=== Step {self.state.global_step} ===")
            print(f"ListNet: {listnet.item():.4f}  Pairwise: {pairwise.item():.4f}  MSE: {mse.item():.4f}")
            print(f"Total Loss: {rank_loss.item():.4f}")
            print(f"Pred scores: {pred_scores.detach().cpu().float().numpy()}")
            print(f"GT scores:   {gt_scores_norm.cpu().float().numpy()}")
            pred_order = torch.argsort(pred_scores, descending=True)
            gt_order   = torch.argsort(gt_scores_norm, descending=True)
            print(f"Pred ranking: {pred_order.cpu().numpy()}")
            print(f"GT ranking:   {gt_order.cpu().numpy()}")
            print(f"Ranking match: {torch.equal(pred_order, gt_order)}")

            # Log attention weight stats to monitor what model focuses on
            print(f"Attn weight std (last batch): "
                  f"{self.attn_pooling.attention.weight.std().item():.6f}")

        return (rank_loss, outputs) if return_outputs else rank_loss

# ── Training args (same as option 1) ──────────────────────────────────────────
training_args = TrainingArguments(
    output_dir=OUTPUT_MODEL,
    overwrite_output_dir=True,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=5,
    learning_rate=1e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    save_total_limit=3,
    fp16=False,
    bf16=False,
    remove_unused_columns=False,
    dataloader_pin_memory=False,
    torch_compile=False,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    max_grad_norm=1.0,
    dataloader_num_workers=0,
)

trainer = AttnPoolingTrainer(
    score_head=model.score_head,
    attn_pooling=model.attn_pooling,
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=val_data,
    data_collator=collate_fn,
)

print("\nStarting Step 2a training (attention pooling + LoRA r=16, ranking loss only)...")
print(f"GPU Memory: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")

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
model.save_pretrained(OUTPUT_MODEL)
processor.save_pretrained(OUTPUT_MODEL)

# Save both score head and attention pooling
torch.save(model.score_head.state_dict(),
           os.path.join(OUTPUT_MODEL, "score_head.pt"))
torch.save(model.attn_pooling.state_dict(),
           os.path.join(OUTPUT_MODEL, "attn_pooling.pt"))
print(f"Score head and attention pooling saved to {OUTPUT_MODEL}/")
print("Done!")