
import os
import re
import shutil

## SET PATHS
data_root = "home/debajyoti/paridhi_mtp/product_images_real"
#data_root = "/home/debajyoti/home/debajyoti/debajyoti/product_images_real"
dataset_response = os.path.join(data_root, "dataset_response_user")  # Original user responses
dataset_response_user = os.path.join(data_root, "dataset_response_user")  # Explicitly define this
second_verification = os.path.join(data_root, "dataset_response_user_2/responses")  # Second verification data
dataset_response_user_2 = os.path.join(data_root, "dataset_response_user_2/responses")  # Second verification data
dataset_response_model = os.path.join(data_root, "dataset_response")  # Original model responses
output_dir = os.path.join(data_root, "final_data")  # New output directory

# Define model mappings
MODEL_NAMES = {
    1: "qwen_zeroshot",
    2: "pixtral_zeroshot",
    3: "qwen_fewshot",
    4: "pixtral_fewshot"
}

MODEL_FILES = {
    "qwen_zeroshot": "output_qwen_zeroshot.txt",
    "qwen_fewshot": "output_qwen_fewshot.txt",
    "pixtral_zeroshot": "output_pixtral_zeroshot.txt",
    "pixtral_fewshot": "output_pixtral_fewshot.txt"
}

def parse_user_preferences(content,group):
    """Parse model preferences from user output file content"""
    preferences = {}
    model_match = re.search(r'model\s*:\s*([^\n]+)', content)
    if model_match:
        model_pref_str = model_match.group(1).strip()
        if "=" in model_pref_str:
            equal_models = [int(m.strip()) for m in model_pref_str.split("=")]
            preferences["top_model"] = equal_models[0]
            preferences["equal_models"] = equal_models
        else:
            try:
                ranked_models = [int(m.strip()) for m in model_pref_str.split(",")]
                preferences["top_model"] = ranked_models[0]
                preferences["ranked_models"] = ranked_models
            except ValueError:
                if model_pref_str.strip():
                    print(f"Invalid model preference {group}: {model_pref_str}")
                    preferences["top_model"] = 1
    return preferences

def process_user_outputs():
    """Process all user outputs and create a new directory with only preferred model outputs"""
    print(f"Processing user outputs and creating directory: {output_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Track statistics
    stats = {
        "total_processed": 0,
        "first_verification": 0,
        "second_verification": 0,
        "errors": 0,
        "model_preferences": {name: 0 for name in MODEL_NAMES.values()},
        "category_stats": {},  # Track model preferences by category
        "first_verification_prefs": {name: 0 for name in MODEL_NAMES.values()},
        "second_verification_prefs": {name: 0 for name in MODEL_NAMES.values()}
    }
    
    # Keep track of processed groups
    processed_groups = set()
    # First, add a print statement to check the second_verification path
    print(f"Second verification path: {second_verification}")
    print(f"Does second verification path exist? {os.path.exists(second_verification)}")

    # If the path exists, list its contents
    if os.path.exists(second_verification):
        print("Contents of second verification path:")
        print(os.listdir(second_verification))

    
    
    # First, process second verification groups (these have higher priority)
    # First, process second verification groups (these have higher priority)
    if os.path.exists(second_verification):
        print("Processing second verification groups...")
        
        # Process each category in the second verification folder
        for category in os.listdir(second_verification):
            category_path = os.path.join(second_verification, category)
            if not os.path.isdir(category_path):
                continue
            
            # Create category folder in output directory
            output_category_dir = os.path.join(output_dir, category)
            os.makedirs(output_category_dir, exist_ok=True)
            
            # Process each group in this category
            for group in os.listdir(category_path):
                group_path = os.path.join(category_path, group)
                if not os.path.isdir(group_path):
                    continue
                
                # Check for real_user_output_2.txt file
                user_output_path = os.path.join(dataset_response_user_2, category, group, "real_user_output_2.txt")
                
                if os.path.exists(user_output_path):
                    try:
                        # Read user preference
                        with open(user_output_path, 'r', encoding='utf-8') as f:
                            user_content = f.read()
                        
                        # Parse preference
                        preferences = parse_user_preferences(user_content, group)
                        
                        if "top_model" not in preferences:
                            print(f"No valid preference found in {category}/{group} (second verification)")
                            stats["errors"] += 1
                            continue
                        
                        top_model_num = preferences["top_model"]
                        top_model_name = MODEL_NAMES.get(top_model_num)
                        
                        if not top_model_name:
                            print(f"Invalid model number {top_model_num} in {category}/{group} (second verification)")
                            stats["errors"] += 1
                            continue
                        
                        # Get corresponding model file
                        model_file = MODEL_FILES.get(top_model_name)
                        model_file_path = os.path.join(dataset_response_model, category, group, model_file)
                        
                        if not os.path.exists(model_file_path):
                            print(f"Model file not found: {model_file_path}")
                            stats["errors"] += 1
                            continue
                        
                        # Create output group directory
                        output_group_dir = os.path.join(output_category_dir, group)
                        os.makedirs(output_group_dir, exist_ok=True)
                        
                        # Copy model output to user_output.txt
                        output_file_path = os.path.join(output_group_dir, "user_output.txt")
                        
                        with open(model_file_path, 'r', encoding='utf-8') as f:
                            model_content = f.read()
                        
                        with open(output_file_path, 'w', encoding='utf-8') as f:
                            f.write(model_content)
                        
                        # Update stats
                        stats["total_processed"] += 1
                        stats["second_verification"] += 1
                        stats["model_preferences"][top_model_name] += 1
                        stats["second_verification_prefs"][top_model_name] += 1
                        
                        # Track category statistics
                        if category not in stats["category_stats"]:
                            stats["category_stats"][category] = {name: 0 for name in MODEL_NAMES.values()}
                        stats["category_stats"][category][top_model_name] += 1
                        
                        # Add to processed groups
                        processed_groups.add(f"{category}/{group}")
                        
                        print(f"Processed second verification group: {category}/{group} - Used {top_model_name}")
                        
                    except Exception as e:
                        print(f"Error processing {category}/{group} (second verification): {str(e)}")
                        stats["errors"] += 1
                        continue
    
    # Next, process first verification groups (skip those already processed)
    print("Processing first verification groups...")
    
    for category in os.listdir(dataset_response):
        category_path = os.path.join(dataset_response, category)
        if not os.path.isdir(category_path):
            continue
        
        # Create category folder in output directory
        output_category_dir = os.path.join(output_dir, category)
        os.makedirs(output_category_dir, exist_ok=True)
        
        # Process each group in this category
        for group in os.listdir(category_path):
            # Skip if already processed in second verification
            if f"{category}/{group}" in processed_groups:
                continue
                
            group_path = os.path.join(category_path, group)
            if not os.path.isdir(group_path):
                continue
            
            # Check for real_user_output.txt file in user responses folder
            user_output_path = os.path.join(dataset_response_user, category, group, "real_user_output.txt")
            
            if os.path.exists(user_output_path):
                try:
                    # Read user preference
                    with open(user_output_path, 'r', encoding='utf-8') as f:
                        user_content = f.read()
                    
                    # Parse preference
                    preferences = parse_user_preferences(user_content,group)
                    
                    if "top_model" not in preferences:
                        print(f"No valid preference found in {category}/{group} (first verification)")
                        stats["errors"] += 1
                        continue
                    
                    top_model_num = preferences["top_model"]
                    top_model_name = MODEL_NAMES.get(top_model_num)
                    
                    if not top_model_name:
                        print(f"Invalid model number {top_model_num} in {category}/{group} (first verification)")
                        stats["errors"] += 1
                        continue
                    
                    # Get corresponding model file
                    model_file = MODEL_FILES.get(top_model_name)
                    model_file_path = os.path.join(dataset_response_model, category, group, model_file)
                    
                    if not os.path.exists(model_file_path):
                        print(f"Model file not found: {model_file_path}")
                        stats["errors"] += 1
                        continue
                    
                    # Create output group directory
                    output_group_dir = os.path.join(output_category_dir, group)
                    os.makedirs(output_group_dir, exist_ok=True)
                    
                    # Copy model output to user_output.txt
                    output_file_path = os.path.join(output_group_dir, "user_output.txt")
                    
                    with open(model_file_path, 'r', encoding='utf-8') as f:
                        model_content = f.read()
                    
                    with open(output_file_path, 'w', encoding='utf-8') as f:
                        f.write(model_content)
                    
                    # Update stats
                    stats["total_processed"] += 1
                    stats["first_verification"] += 1
                    stats["model_preferences"][top_model_name] += 1
                    stats["first_verification_prefs"][top_model_name] += 1
                    
                    # Track category statistics
                    if category not in stats["category_stats"]:
                        stats["category_stats"][category] = {name: 0 for name in MODEL_NAMES.values()}
                    stats["category_stats"][category][top_model_name] += 1
                    
                    print(f"Processed first verification group: {category}/{group} - Used {top_model_name}")
                    
                except Exception as e:
                    print(f"Error processing {category}/{group} (first verification): {str(e)}")
                    stats["errors"] += 1
                    continue
    
    # Calculate detailed statistics
    total_valid = sum(stats["model_preferences"].values())
    model_percentages = {model: (count/total_valid*100 if total_valid > 0 else 0) 
                         for model, count in stats["model_preferences"].items()}
    
    # Print summary
    print("\nProcessing complete!")
    print(f"Total processed: {stats['total_processed']}")
    print(f"First verification: {stats['first_verification']}")
    print(f"Second verification: {stats['second_verification']}")
    print(f"Errors: {stats['errors']}")
    
    print("\n=== MODEL PREFERENCE STATISTICS ===")
    print(f"Total valid responses: {total_valid}")
    print("\nModel preferences (overall):")
    for model_name, count in sorted(stats["model_preferences"].items(), key=lambda x: x[1], reverse=True):
        percentage = model_percentages[model_name]
        print(f"  {model_name}: {count} ({percentage:.2f}%)")
    
    # Save summary to output directory
    with open(os.path.join(output_dir, "model_statistics.txt"), 'w', encoding='utf-8') as f:
        f.write("MODEL PREFERENCE STATISTICS\n")
        f.write("==========================\n\n")
        f.write(f"Total valid responses: {total_valid}\n")
        f.write(f"First verification: {stats['first_verification']}\n")
        f.write(f"Second verification: {stats['second_verification']}\n")
        f.write(f"Errors: {stats['errors']}\n\n")
        
        f.write("MODEL PREFERENCES (OVERALL)\n")
        f.write("--------------------------\n")
        for model_name, count in sorted(stats["model_preferences"].items(), key=lambda x: x[1], reverse=True):
            percentage = model_percentages[model_name]
            f.write(f"{model_name}: {count} ({percentage:.2f}%)\n")
        
        f.write("\nZERO-SHOT VS FEW-SHOT COMPARISON\n")
        f.write("-------------------------------\n")
        zero_shot_total = stats["model_preferences"]["qwen_zeroshot"] + stats["model_preferences"]["pixtral_zeroshot"]
        few_shot_total = stats["model_preferences"]["qwen_fewshot"] + stats["model_preferences"]["pixtral_fewshot"]
        zero_shot_pct = (zero_shot_total/total_valid*100) if total_valid > 0 else 0
        few_shot_pct = (few_shot_total/total_valid*100) if total_valid > 0 else 0
        f.write(f"Zero-shot approaches: {zero_shot_total} ({zero_shot_pct:.2f}%)\n")
        f.write(f"Few-shot approaches: {few_shot_total} ({few_shot_pct:.2f}%)\n\n")
        
        f.write("MODEL COMPARISON\n")
        f.write("---------------\n")
        qwen_total = stats["model_preferences"]["qwen_zeroshot"] + stats["model_preferences"]["qwen_fewshot"]
        pixtral_total = stats["model_preferences"]["pixtral_zeroshot"] + stats["model_preferences"]["pixtral_fewshot"]
        qwen_pct = (qwen_total/total_valid*100) if total_valid > 0 else 0
        pixtral_pct = (pixtral_total/total_valid*100) if total_valid > 0 else 0
        f.write(f"Qwen models (total): {qwen_total} ({qwen_pct:.2f}%)\n")
        f.write(f"Pixtral models (total): {pixtral_total} ({pixtral_pct:.2f}%)\n\n")
        
        f.write("VERIFICATION COMPARISON\n")
        f.write("----------------------\n")
        f.write("First verification preferences:\n")
        first_total = sum(stats["first_verification_prefs"].values())
        for model_name, count in sorted(stats["first_verification_prefs"].items(), key=lambda x: x[1], reverse=True):
            pct = (count/first_total*100) if first_total > 0 else 0
            f.write(f"  {model_name}: {count} ({pct:.2f}%)\n")
            
        f.write("\nSecond verification preferences:\n")
        second_total = sum(stats["second_verification_prefs"].values())
        for model_name, count in sorted(stats["second_verification_prefs"].items(), key=lambda x: x[1], reverse=True):
            pct = (count/second_total*100) if second_total > 0 else 0
            f.write(f"  {model_name}: {count} ({pct:.2f}%)\n")
        
        f.write("\nCATEGORY-WISE MODEL PREFERENCES\n")
        f.write("------------------------------\n")
        for category, model_counts in sorted(stats["category_stats"].items()):
            f.write(f"\n{category.upper()}:\n")
            category_total = sum(model_counts.values())
            for model_name, count in sorted(model_counts.items(), key=lambda x: x[1], reverse=True):
                pct = (count/category_total*100) if category_total > 0 else 0
                f.write(f"  {model_name}: {count} ({pct:.2f}%)\n")
    
    return stats

if __name__ == "__main__":
    process_user_outputs()