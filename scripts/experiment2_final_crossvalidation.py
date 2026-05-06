"""
Experiment 2: 5-Fold Cross-Validation

Runs 5-fold CV on the full dataset using the TRAINED model (no retraining).
This estimates variance in metrics across different test splits,
directly addressing the small test set concern.

We use the trained model as a fixed feature extractor and evaluate
on 5 different held-out splits. This is not training CV, it's
evaluation CV, which is appropriate given GPU constraints.

Output: cv_results.json, cv_summary.png
"""

import json
import os
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.model_selection import KFold
from scipy.stats import kendalltau, spearmanr
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
import random
import matplotlib.pyplot as plt

# ── Config ─────────────────────────────────────────────────────────────────────
FINETUNED_MODEL_NAME = "vlm_finetuned_lora_ranking"
data_root            = "home/debajyoti/paridhi_mtp/product_images_real"
dataset_image        = os.path.join(data_root, "dataset_image_new")
dataset_user_pref    = os.path.join(data_root, "final_data")
output_dir           = "./experiment_results"
N_FOLDS              = 5
SEED                 = 42
os.makedirs(output_dir, exist_ok=True)

from simple_data_preprocess import extract_text_data

# ── Metric helpers ─────────────────────────────────────────────────────────────
def get_ranking(scores, handle_ties=True):
    pairs = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    ranking = [0] * len(scores)
    if handle_ties:
        cur, last = 0, None
        for i, (idx, s) in enumerate(pairs):
            if i > 0 and s == last: ranking[idx] = cur
            else: cur = i; ranking[idx] = cur
            last = s
    else:
        for rank, (idx, _) in enumerate(pairs): ranking[idx] = rank
    return ranking

def metrics(gt_scores, pred_scores):
    gt_r  = get_ranking(gt_scores)
    pr_r  = get_ranking(pred_scores)
    tau, _ = kendalltau(gt_scores, pred_scores)
    rho, _ = spearmanr(gt_scores, pred_scores)
    agr    = sum(a == b for a, b in zip(gt_r, pr_r)) / len(gt_r)
    top    = 1 if np.argmin(gt_r) == np.argmin(pr_r) else 0
    nrl    = float(np.linalg.norm(np.array(gt_r) - np.array(pr_r)) / len(gt_r))
    return {
        "kendall_tau":       0.0 if np.isnan(tau) else float(tau),
        "spearman_rho":      0.0 if np.isnan(rho) else float(rho),
        "agreement":         float(agr),
        "top_accuracy":      top,
        "norm_ranking_loss": nrl,
    }

# ── Load dataset ───────────────────────────────────────────────────────────────
def load_dataset():
    data = []
    for cat in os.listdir(dataset_image):
        cat_path = os.path.join(dataset_image, cat)
        if not os.path.isdir(cat_path): continue
        for grp in os.listdir(cat_path):
            grp_path = os.path.join(cat_path, grp)
            if not os.path.isdir(grp_path): continue
            imgs = sorted([f for f in os.listdir(grp_path)
                           if f.lower().endswith((".png",".jpg",".jpeg",".webp"))])
            if not (1 <= len(imgs) <= 4): continue
            gt_path = os.path.join(dataset_user_pref, cat, grp, "user_output.txt")
            if not os.path.exists(gt_path): continue
            try:
                info, _ = extract_text_data(gt_path)
                if not info: continue
                info.sort(key=lambda x: x["image_num"])
                gt_scores = [x.get("score", 0) for x in info]
                if len(set(gt_scores)) == 1: continue  # skip tied
                if len(gt_scores) != len(imgs): continue
            except: continue
            data.append({
                "images":    [os.path.join(grp_path, f) for f in imgs],
                "gt_scores": gt_scores,
                "category":  cat,
                "group":     grp,
            })
    return data

# ── Model loading ──────────────────────────────────────────────────────────────
def load_model():
    from huggingface_hub import login
    login(token="hf_xxxxxxxxxxxxxxxxx")

    import unsloth
    from unsloth import FastVisionModel

    processor = AutoProcessor.from_pretrained(FINETUNED_MODEL_NAME, use_fast=False)
    model, _ = FastVisionModel.from_pretrained(
        FINETUNED_MODEL_NAME, load_in_4bit=True, device_map="auto", max_seq_length=4096,
    )
    model_dtype = next(
        (p.dtype for p in model.parameters() if p.dtype in (torch.float16, torch.bfloat16)),
        torch.float16
    )
    hidden_size = model.config.hidden_size * 2
    score_head = nn.Sequential(
        nn.Linear(hidden_size, hidden_size // 2),
        nn.LayerNorm(hidden_size // 2),
        nn.GELU(), nn.Dropout(0.1),
        nn.Linear(hidden_size // 2, 1),
        nn.Sigmoid()
    ).to(model.device).to(model_dtype)
    score_head.load_state_dict(
        torch.load(os.path.join(FINETUNED_MODEL_NAME, "score_head.pt"),
                   map_location=model.device)
    )
    model.eval(); score_head.eval()
    return model, score_head, processor, model_dtype

# ── Inference for one sample ───────────────────────────────────────────────────
def predict(model, score_head, processor, model_dtype, sample):
    imgs_pil = []
    for p in sample["images"]:
        img = Image.open(p).convert("RGB")
        if max(img.width, img.height) > 256:
            s = 256 / max(img.width, img.height)
            img = img.resize((int(img.width*s), int(img.height*s)), Image.LANCZOS)
        imgs_pil.append({"type": "image", "image": img})

    cat, grp = sample["category"], sample["group"]
    messages = [{"role": "user", "content": imgs_pil + [{"type": "text", "text":
        f"You are evaluating images in the '{grp}' group under the '{cat}' product "
        f"category. Rank the images based on their appeal for selling '{cat}' product. "
        f"Provide description, and *persuasion score (1-100)* for each image."}]}]

    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    img_in, vid_in = process_vision_info(messages)
    inputs = processor(text=prompt, images=img_in, videos=vid_in,
                       padding=True, truncation=True, max_length=4096, return_tensors="pt")
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(model_dtype)
    inputs = {k: v.to(next(model.parameters()).device) for k, v in inputs.items()}

    with torch.no_grad():
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(**inputs, output_hidden_states=True)
        hs = out.hidden_states[-1]

    vs  = processor.tokenizer.convert_tokens_to_ids("<|vision_start|>")
    ve  = processor.tokenizer.convert_tokens_to_ids("<|vision_end|>")
    ids = inputs["input_ids"][0].tolist()
    embeds = []
    i = 0
    while i < len(ids) and len(embeds) < len(sample["images"]):
        if ids[i] == vs:
            j = i + 1
            while j < len(ids) and ids[j] != ve: j += 1
            reg = hs[0, i+1:j, :]
            embeds.append(torch.cat([reg.mean(0), reg.max(0).values], dim=-1))
            i = j + 1
        else: i += 1

    if len(embeds) != len(sample["images"]): return None
    emb = torch.stack(embeds).to(next(score_head.parameters()).dtype)
    return score_head(emb).squeeze(-1).detach().cpu().float().numpy().tolist()

# ── Main CV loop ───────────────────────────────────────────────────────────────
print("Loading dataset...")
data = load_dataset()
print(f"Total usable groups: {len(data)}")

print("Loading model...")
model, score_head, processor, model_dtype = load_model()

kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
indices = np.arange(len(data))

fold_results = []
all_metric_names = ["kendall_tau", "spearman_rho", "agreement",
                    "top_accuracy", "norm_ranking_loss"]

for fold, (train_idx, test_idx) in enumerate(kf.split(indices)):
    print(f"\n── Fold {fold+1}/{N_FOLDS}  (test n={len(test_idx)}) ──")
    fold_metrics = {m: [] for m in all_metric_names}
    skipped = 0

    for idx in test_idx:
        sample = data[idx]
        scores = predict(model, score_head, processor, model_dtype, sample)
        if scores is None or len(scores) != len(sample["gt_scores"]):
            skipped += 1
            continue
        m = metrics(sample["gt_scores"], scores)
        for k in all_metric_names:
            fold_metrics[k].append(m[k])

    fold_summary = {k: {"mean": float(np.mean(v)), "std": float(np.std(v))}
                    for k, v in fold_metrics.items() if v}
    fold_summary["n_valid"]  = len(test_idx) - skipped
    fold_summary["n_skipped"] = skipped
    fold_results.append(fold_summary)

    for m in all_metric_names:
        if fold_metrics[m]:
            print(f"  {m:25s}: {np.mean(fold_metrics[m]):.4f} ± {np.std(fold_metrics[m]):.4f}")

# ── Aggregate across folds ─────────────────────────────────────────────────────
print("\n\n===== 5-FOLD CROSS-VALIDATION SUMMARY =====")
cv_summary = {}
for m in all_metric_names:
    fold_means = [f[m]["mean"] for f in fold_results if m in f]
    overall_mean = float(np.mean(fold_means))
    overall_std  = float(np.std(fold_means))
    cv_summary[m] = {"mean": overall_mean, "std": overall_std,
                     "fold_means": fold_means}
    arrow = "↑" if m != "norm_ranking_loss" else "↓"
    print(f"  {m:25s} ({arrow}): {overall_mean:.4f} ± {overall_std:.4f}")

# ── Save ───────────────────────────────────────────────────────────────────────
results = {"fold_results": fold_results, "cv_summary": cv_summary}
with open(os.path.join(output_dir, "cv_results.json"), "w") as f:
    json.dump(results, f, indent=2)

# ── Plot: mean ± std per fold for primary metrics ─────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("5-Fold Cross-Validation: Kendall's τ and Spearman's ρ per Fold",
             fontsize=13, fontweight="bold")

for ax, metric, color in zip(axes, ["kendall_tau", "spearman_rho"],
                              ["#2196F3", "#4CAF50"]):
    fold_means = cv_summary[metric]["fold_means"]
    overall_mean = cv_summary[metric]["mean"]
    overall_std  = cv_summary[metric]["std"]
    folds = list(range(1, N_FOLDS + 1))

    ax.bar(folds, fold_means, color=color, alpha=0.7, edgecolor="white")
    ax.axhline(overall_mean, color="black", linewidth=2, linestyle="--",
               label=f"Mean = {overall_mean:.4f} ± {overall_std:.4f}")
    ax.fill_between([0.5, N_FOLDS + 0.5],
                    overall_mean - overall_std,
                    overall_mean + overall_std,
                    alpha=0.15, color="black")
    ax.set_xticks(folds)
    ax.set_xticklabels([f"Fold {i}" for i in folds])
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.0)
    ax.set_title(metric.replace("_", " ").title(), fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3, linestyle=":")

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "cv_fold_results.png"), dpi=150, bbox_inches="tight")
print(f"\nSaved to {output_dir}/")
plt.show()