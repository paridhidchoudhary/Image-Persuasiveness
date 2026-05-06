import os
from simple_data_preprocess import extract_text_data

data_root = "home/debajyoti/paridhi_mtp/product_images_real"
dataset_image = os.path.join(data_root, "dataset_image_new")
dataset_response = os.path.join(data_root, "final_data")
MAX_IMAGES = 4

print("Checking if TRAINING and EVAL load data the same way...\n")

# Simulate TRAINING data loading
train_data_sim = []
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
        gt_scores = []
        for img_info in extracted_info:
            gt_scores.append(img_info.get("score", 0))
        
        # TRAINING FILTER
        if len(gt_scores) == 0 or all(s == 0 for s in gt_scores):
            continue
        
        train_data_sim.append({
            "category": category,
            "group": group
        })

print(f"TRAINING-style loading: {len(train_data_sim)} groups\n")

# Simulate EVAL data loading (check your eval code for differences)
eval_data_sim = []
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
        
        # Check if using different path?
        user_preferred_path = os.path.join(dataset_response, category, group, "user_output.txt")
        
        if not os.path.exists(user_preferred_path):
            continue
        
        # Does eval extract scores differently?
        with open(user_preferred_path, 'r') as f:
            content = f.read()
        
        extracted_info, ranking = extract_text_data(user_preferred_path)
        scores = [item.get("score", 0) for item in extracted_info]
        
        # Check if eval has different filtering
        if len(scores) == 0:
            continue
        
        eval_data_sim.append({
            "category": category,
            "group": group
        })

print(f"EVAL-style loading: {len(eval_data_sim)} groups\n")

if len(train_data_sim) != len(eval_data_sim):
    print("⚠️ MISMATCH! Training and eval load different numbers of groups")
    train_set = set((d['category'], d['group']) for d in train_data_sim)
    eval_set = set((d['category'], d['group']) for d in eval_data_sim)
    
    in_train_not_eval = train_set - eval_set
    in_eval_not_train = eval_set - train_set
    
    if in_train_not_eval:
        print(f"\nIn TRAINING but not EVAL ({len(in_train_not_eval)} groups):")
        for cat, grp in sorted(in_train_not_eval)[:10]:
            print(f"  {cat}/{grp}")
    
    if in_eval_not_train:
        print(f"\nIn EVAL but not TRAINING ({len(in_eval_not_train)} groups):")
        for cat, grp in sorted(in_eval_not_train)[:10]:
            print(f"  {cat}/{grp}")
else:
    print("✓ Training and eval load the same groups")