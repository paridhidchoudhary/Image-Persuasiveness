# Persuasive Image Ranking using Vision-Language Models

This project explores how **small visual changes affect image persuasiveness** for specific product categories, and builds a pipeline to **train and evaluate models that rank images based on persuasiveness**.

---

## Resources

* **Final Model (LoRA Score Head + Explanations)**
  https://huggingface.co/paridhidchoudhary/persuasort-qwen-adapter

* **Dataset (Persuasort Dataset)**
  https://huggingface.co/datasets/paridhidchoudhary/persuasort-dataset

---

## Problem Statement

Given a set of images of a product, the goal is to:

> Identify the **most persuasive image** for that product category.

We study how **minimal edits** (inspired by VISMIN) influence persuasiveness and train models to capture this.

---

## Repository Structure

* `scripts/` → All training and evaluation pipelines, has a Readme too to describe overall method
* `experiment_results/` → Outputs, statistical analysis, plots
* `MUSIQ/` → MUSIQ baseline implementation
* `nima_baseline_results/` → NIMA baseline outputs
* `vismin_dataset/` → Processed VISMIN-based dataset

---

## Dataset Creation Pipeline

We build upon the VISMIN dataset:
https://arxiv.org/pdf/2407.16772

### Steps:

1. **Grouping**

   * Group images by original + edited variants

2. **Category Assignment**

   * Categories (12 total):

     * Electronics: Laptop, TV, Phone, Remote
     * Kitchen: Microwave, Toaster, Refrigerator
     * Luggage: Suitcase, Handbag, Backpack
     * Furniture: Chair, Couch

3. **Sampling**

   * 50 samples per category for benchmark dataset

4. **Model Outputs**

   * Models used:

     * Qwen-2.5-7B-VL
     * Pixtral-Large
   * Settings:

     * Zero-shot
     * Few-shot

5. **Human Evaluation**

   * Annotators select best output
   * If none are correct → human-generated answer used

6. **Bias Mitigation**

   * Low agreement samples re-evaluated by additional annotators


## Experiments

* Statistical analysis (boxplots, hypothesis testing)
* Cross-validation
* Score distribution analysis
* Group size sensitivity

---

## Baselines

### NIMA (Neural Image Assessment)

https://github.com/titu1994/neural-image-assessment

### MUSIQ

Implemented in `MUSIQ/`

---


## Authors

Paridhi Choudhary
IIT Kharagpur

---

## Summary

This project:

* Builds a **persuasiveness-aware dataset**
* Incorporates **human feedback**
* Trains **ranking-aware VLMs**
* Produces **interpretable outputs (with explanations)**

---
