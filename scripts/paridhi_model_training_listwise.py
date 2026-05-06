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

login(token="hf_xxxxxxxxxxxxxx")

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

data_root = "home/debajyoti/paridhi_mtp/product_images_real"
dataset_image = os.path.join(data_root, "dataset_image_new")
dataset_response = os.path.join(data_root, "final_data")
GLOBAL_MIN, GLOBAL_MAX = get_global_normalization(dataset_response)
output_model = "./vlm_finetuned_listwise"
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
hidden_size = model.config.hidden_size*2
# model.score_head = nn.Linear(hidden_size, 1).to(model.device)

model.score_head = nn.Sequential(
    nn.Linear(hidden_size, hidden_size // 2),
    nn.LayerNorm(hidden_size // 2),
    nn.GELU(),
    nn.Dropout(0.1),
    nn.Linear(hidden_size // 2, 1),
    nn.Sigmoid()  # Forces output to [0, 1]
).to(model.device).to(torch.bfloat16)

# Initialize weights
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

model.score_head.apply(init_weights)
# Freeze everything in the PEFT model
for param in model.parameters():
    param.requires_grad = False

# score_head is separate, its params are trainable by default
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
                    gt_scores = []
                    for img_info in extracted_info:
                        gt_scores.append(img_info.get("score", 0))
                    
                    data.append({
                        "images": images,
                        "full_response_text": full_response_text,
                        "ranking": ranking,
                        "scores": gt_scores,
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
        
        # labels = processor(
        #     text=full_text,
        #     padding=True,
        #     truncation=True,
        #     max_length=2048,
        #     return_tensors="pt"
        # )["input_ids"]
        
        input_len = inputs["input_ids"].shape[1]
        # labels_len = labels.shape[1]
        
        # if labels_len < input_len:
        #     padding = torch.full((1, input_len - labels_len), -100, dtype=torch.long)
        #     labels = torch.cat([labels, padding], dim=-1)
        # elif labels_len > input_len:
        #     labels = labels[:, :input_len]
            
        # inputs["labels"] = torch.where(
        #     inputs["input_ids"] == processor.tokenizer.pad_token_id,
        #     -100,
        #     labels
        # )
        
        inputs["num_items_in_batch"] = torch.tensor([len(batch["images"])], dtype=torch.long)
        
        # Check if scores exist and are not empty
        if "scores" not in batch or len(batch["scores"]) == 0:
            print(f"Warning: No scores for {category}/{group}, using dummy scores")
            gt_scores = torch.zeros(len(batch["images"]), dtype=torch.float)
        else:
            gt_scores = torch.tensor(batch["scores"], dtype=torch.float)
            
            # Only normalize if there are multiple different values
            if gt_scores.numel() > 0:
                gt_scores = (gt_scores - GLOBAL_MIN) / (GLOBAL_MAX - GLOBAL_MIN)
        
        inputs["gt_scores"] = gt_scores
        
        return inputs
        
    except Exception as e:
        print(f"Error in collate: {e}")
        import traceback
        traceback.print_exc()
        torch.cuda.empty_cache()
        gc.collect()
        
        # Return dummy data WITH proper gt_scores
        return {
            "input_ids": torch.tensor([[0]]), 
            "attention_mask": torch.tensor([[0]]), 
            "labels": torch.tensor([[-100]]),
            "num_items_in_batch": torch.tensor([1], dtype=torch.long),
            "gt_scores": torch.tensor([0.5], dtype=torch.float)  # Use 0.5 as neutral score
        }

def listnet_loss(pred_scores, gt_scores):
    """
    ListNet loss with KL divergence
    Both pred_scores and gt_scores should be in [0, 1] range
    """
    # Convert to probability distributions
    P_pred = torch.softmax(pred_scores, dim=0)
    P_gt = torch.softmax(gt_scores, dim=0)
    
    # Cross-entropy formulation (more stable)
    loss = -torch.sum(P_gt * torch.log(P_pred + 1e-10))
    return loss


def pairwise_margin_loss(pred_scores, gt_scores, margin=0.1):  # ← Reduce margin for [0,1] range
    """
    Pairwise ranking with margin
    Margin should be small since scores are in [0, 1]
    """
    losses = []
    n = len(pred_scores)
    
    for i in range(n):
        for j in range(i+1, n):
            if gt_scores[i] > gt_scores[j]:
                # pred_scores[i] should be > pred_scores[j]
                loss = torch.clamp(margin - (pred_scores[i] - pred_scores[j]), min=0)
                losses.append(loss)
            elif gt_scores[j] > gt_scores[i]:
                loss = torch.clamp(margin - (pred_scores[j] - pred_scores[i]), min=0)
                losses.append(loss)
    
    if len(losses) == 0:
        return torch.tensor(0.0, device=pred_scores.device)
    
    return torch.mean(torch.stack(losses))


def mse_loss_normalized(pred_scores, gt_scores):
    """
    Simple MSE loss - works well when both are [0, 1]
    """
    return torch.mean((pred_scores - gt_scores) ** 2)
    
class CustomTrainer(Trainer):
    def __init__(self, score_head, **kwargs):
        super().__init__(**kwargs)
        self.score_head = score_head

    def create_optimizer(self):
        self.optimizer = torch.optim.AdamW(
            self.score_head.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
        )
        return self.optimizer

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # ---- Pop extra inputs with safety ----
        inputs.pop("num_items_in_batch", None)
        gt_scores = inputs.pop("gt_scores", None)
        inputs.pop("labels", None)
        #labels = inputs.pop("labels", None)
        
        # ---- Safety checks ----
        if gt_scores is None:
            print("Warning: Missing gt_scores, skipping batch")
            return torch.tensor(0.0, device=model.device, requires_grad=True)
        
        if gt_scores.shape[0] == 1 and gt_scores[0] == 0.0:
            print("Warning: Detected dummy batch, skipping")
            return torch.tensor(0.0, device=model.device, requires_grad=True)
        
        # ---- Forward pass ----
        try:
            outputs = model(**inputs, output_hidden_states=True)
            logits = outputs.logits
            hidden_states = outputs.hidden_states[-1]
        except Exception as e:
            print(f"Error in forward pass: {e}")
            return torch.tensor(0.0, device=model.device, requires_grad=True)
        
        # ---- LM Loss ----
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        lm_loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        lm_loss = lm_loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )
        
        # ---- Ranking Loss ----
        num_images = gt_scores.shape[0]

        # Extract from vision tokens, not first N tokens
        # vision_start_id = 151652
        # vision_end_id = 151653
        vision_start_id = processor.tokenizer.convert_tokens_to_ids("<|vision_start|>")
        vision_end_id = processor.tokenizer.convert_tokens_to_ids("<|vision_end|>")
        
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
                
                # image_region = hidden_states[0, start:end, :]
                # pooled = image_region.mean(dim=0)

                image_region = hidden_states[0, start+1:end-1, :]

                mean_pool = image_region.mean(dim=0)
                max_pool = image_region.max(dim=0).values

                pooled = torch.cat([mean_pool, max_pool], dim=-1).to(torch.bfloat16)
                image_embeddings.append(pooled)
                
                i = end
            else:
                i += 1
        
        if len(image_embeddings) != num_images:
            print(f"Warning: Could not extract {num_images} images, using LM loss only")
            return (lm_loss, outputs) if return_outputs else lm_loss
        
        image_embeds = torch.stack(image_embeddings)
        #pred_scores = model.score_head(image_embeds).squeeze(-1)
        
        pred_scores = self.score_head(image_embeds).squeeze(-1)

        gt_scores_norm = gt_scores.to(pred_scores.device).to(torch.bfloat16)
        
        # Three ranking losses
        listnet = listnet_loss(pred_scores, gt_scores_norm)
        pairwise = pairwise_margin_loss(pred_scores, gt_scores_norm, margin=0.1)
        mse = mse_loss_normalized(pred_scores, gt_scores_norm)
        
        # Combine ranking losses with emphasis on MSE for score accuracy
        rank_loss = 0.3 * listnet + 0.2 * pairwise + 0.5 * mse
        
        # Final combined loss - almost pure ranking
        # loss = 0.2 * lm_loss + 0.8 * rank_loss
        loss = rank_loss        
        # ---- Debugging ----
        if self.state.global_step % 50 == 0:
            print(f"\n=== Step {self.state.global_step} ===")
            print(f"LM Loss: {lm_loss.item():.4f}")
            print(f"ListNet: {listnet.item():.4f}, Pairwise: {pairwise.item():.4f}, MSE: {mse.item():.4f}")
            print(f"Total Rank Loss: {rank_loss.item():.4f}")
            print(f"Total Loss: {loss.item():.4f}")
            print(f"Pred scores: {pred_scores.detach().cpu().numpy()}")
            print(f"GT scores: {gt_scores_norm.cpu().numpy()}")
            print(f"Score diff (MSE per item): {(pred_scores - gt_scores_norm).pow(2).detach().cpu().numpy()}")
            
            pred_order = torch.argsort(pred_scores, descending=True)
            gt_order = torch.argsort(gt_scores_norm, descending=True)
            print(f"Pred ranking: {pred_order.cpu().numpy()}")
            print(f"GT ranking: {gt_order.cpu().numpy()}")
            print(f"Ranking match: {torch.equal(pred_order, gt_order)}")
        
        return (loss, outputs) if return_outputs else loss

training_args = TrainingArguments(
    output_dir=output_model,
    overwrite_output_dir = True,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=10,
    learning_rate=5e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.2,
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    logging_steps=10,
    load_best_model_at_end = True,
    metric_for_best_model = "eval_loss",
    greater_is_better = False,
    save_total_limit = 3,
    fp16=False,
    remove_unused_columns=False,
    dataloader_pin_memory=False,
    torch_compile=False,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    #optim="paged_adamw_8bit",
    max_grad_norm=1.0,
    dataloader_num_workers=0,
)

# Use CustomTrainer instead of Trainer
trainer = CustomTrainer(
    score_head = model.score_head,
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

# Save score_head separately
score_head_path = os.path.join(output_model, "score_head.pt")
torch.save(model.score_head.state_dict(), score_head_path)
print(f"Score head saved to {score_head_path}")

print("Done!")