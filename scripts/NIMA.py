import os
import subprocess
import re
import json
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from collections import defaultdict

# Import your existing preprocessing function
from simple_data_preprocess import extract_text_data

# --- Configuration ---
# Path to the directory containing the evaluate_mobilenet.py script

## SET PATHS
SCRIPT_DIR = '/home/deepg/Persuasive_Image/neural-image-assessment'

# Name of the evaluation script
EVAL_SCRIPT_NAME = 'evaluate_mobilenet.py'

# Directory where you want to save the output files
OUTPUT_BASE_DIR = '/home/deepg/Persuasive_Image/nima_results'

# Directory for your dataset images
DATA_ROOT = '/home/deepg/NAS/Downloads/MTP-2-persuasion/dataset'
DATASET_IMAGE_DIR = os.path.join(DATA_ROOT, 'dataset_image')
DATASET_RESPONSE_DIR = os.path.join(DATA_ROOT, 'final_data')

# Output directory for raw outputs and debug information
RAW_OUTPUT_DIR = os.path.join(OUTPUT_BASE_DIR, 'raw_outputs')

# Train/val split parameters - use same seed as your model training
RANDOM_SEED = 50  # Use the same random_state as in your model training
TEST_SIZE = 0.05  # Use same test_size as in your model training

# Optional: Set to True if you want to use the -resize true argument
RESIZE_IMAGES = True

# --- Helper Functions ---

def parse_nima_output(output_text):
    """Parse NIMA output to extract scores and ranking"""
    scores = {}
    image_nums = {}
    
    # Extract evaluations and scores
    eval_pattern = r'Evaluating\s+:\s+(.+?)\nNIMA Score\s+:\s+(\d+\.\d+)'
    evaluations = re.findall(eval_pattern, output_text)
    
    if evaluations:
        for image_path, score in evaluations:
            # Extract filename from path
            filename = os.path.basename(image_path)
            scores[filename] = float(score)
            
            # Determine image number
            if 'original' in filename:
                # Original images are typically the highest numbered
                image_nums[filename] = 999  # placeholder, will be adjusted later
            elif 'edit' in filename:
                # Try to extract edit number
                match = re.search(r'edit_(\d+)', filename)
                if match:
                    image_nums[filename] = int(match.group(1))
                else:
                    image_nums[filename] = 998  # fallback for edit without number
    
    # Adjust original image number to be the highest
    if image_nums:
        max_edit_num = max([num for num in image_nums.values() if num < 900], default=0)
        for filename in scores.keys():
            if 'original' in filename:
                image_nums[filename] = max_edit_num + 1
    
    # Create standardized image data format (similar to extract_text_data output)
    image_data = []
    for filename, score in scores.items():
        # Find the image number
        image_num = image_nums.get(filename, 0)
        if image_num == 999:  # Original without proper number assignment
            continue
            
        image_data.append({
            'image_num': image_num,
            'score': score,
            'full_text': f"Image {image_num}: NIMA aesthetic score: {score}",
            'filename': filename
        })
    
    # Sort by image number
    image_data.sort(key=lambda x: x['image_num'])
    
    # Create a ranking text similar to your preprocessing format
    ranking_text = "###Ranking:\n"
    # Sort filenames by scores (descending)
    sorted_by_score = sorted([(filename, scores[filename]) for filename in scores.keys()], 
                            key=lambda x: x[1], reverse=True)
    
    for rank, (filename, score) in enumerate(sorted_by_score, 1):
        ranking_text += f"{rank}. **{filename}** - Score: {score}\n"
    
    return image_data, ranking_text

def get_ranking_from_scores(image_data):
    """Convert scores to rankings (higher score = better rank)"""
    if not image_data:
        return []
    
    # Sort by score (descending)
    sorted_items = sorted(image_data, key=lambda x: x.get('score', 0), reverse=True)
    
    # Create ranking list using image numbers
    return [item['image_num'] for item in sorted_items]

def calculate_rank_agreement(ranking1, ranking2):
    """Calculate agreement between two rankings"""
    if not ranking1 or not ranking2 or len(set(ranking1).intersection(set(ranking2))) < 2:
        return 0
    
    # Create positional maps for both rankings
    map1 = {item: i for i, item in enumerate(ranking1)}
    map2 = {item: i for i, item in enumerate(ranking2)}
    
    # Find common items
    common_items = set(map1.keys()).intersection(set(map2.keys()))
    
    if len(common_items) < 2:
        return 0
    
    # Count positions where ranking agrees
    agreements = 0
    total_comparisons = 0
    
    for item1 in common_items:
        for item2 in common_items:
            if item1 != item2:
                total_comparisons += 1
                # Check if relative ordering is the same
                if (map1[item1] < map1[item2] and map2[item1] < map2[item2]) or \
                   (map1[item1] > map1[item2] and map2[item1] > map2[item2]):
                    agreements += 1
    
    if total_comparisons == 0:
        return 0
        
    return agreements / total_comparisons

# --- Main Script ---

# Create output directories
os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
results_dir = os.path.join(OUTPUT_BASE_DIR, 'analysis')
os.makedirs(results_dir, exist_ok=True)
os.makedirs(RAW_OUTPUT_DIR, exist_ok=True)

# Full path to the evaluation script
eval_script_path = os.path.join(SCRIPT_DIR, EVAL_SCRIPT_NAME)

# Collect all image groups for dataset creation
all_data = []
categories = os.listdir(DATASET_IMAGE_DIR)
for category in categories:
    category_path = os.path.join(DATASET_IMAGE_DIR, category)
    if os.path.isdir(category_path):
        for group in os.listdir(category_path):
            group_path = os.path.join(category_path, group)
            if os.path.isdir(group_path):
                # Check if we have images and a corresponding response
                images = [img for img in os.listdir(group_path) 
                         if img.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))]
                
                if len(images) <= 4 and len(images) > 0:  # Use same MAX_IMAGES=4 constraint as in model training
                    response_path = os.path.join(DATASET_RESPONSE_DIR, category, group, "user_output.txt")
                    if os.path.exists(response_path):
                        all_data.append({
                            'category': category,
                            'group': group,
                            'image_path': group_path,
                            'response_path': response_path
                        })

# Split the dataset using the SAME random state as your model training
print(f"Total dataset size: {len(all_data)} groups")
train_data, test_data = train_test_split(all_data, test_size=TEST_SIZE, random_state=RANDOM_SEED)
print(f"Evaluation set size: {len(test_data)} groups")

# Use only the evaluation set groups
image_groups = test_data

print(f"Starting NIMA evaluation for {len(image_groups)} image groups...")

# Prepare to store results
results = {
    'groups': [],
    'summary': {
        'total_groups': 0,
        'successful_evaluations': 0,
        'average_nima_user_agreement': 0,
        'top_rank_accuracy': 0
    }
}

# Create vectors to store match results
agreements = []
agreement_vector = []  # Will store all agreement values
top_rank_matches = 0
top_rank_vector = []   # Will store 1 for match, 0 for no match
total_successful = 0
errors = []

# Evaluate each group
for group_info in tqdm(image_groups, desc="Evaluating image groups"):
    category = group_info['category']
    group = group_info['group']
    group_dir = group_info['image_path']
    response_path = group_info['response_path']
    
    group_result = {
        'category': category,
        'group': group,
        'success': False
    }
    
    # Create output file name
    output_file = os.path.join(RAW_OUTPUT_DIR, f'{category}_{group}_nima_output.txt')
    
    # Build command
    command = [
        'python3', eval_script_path, 
        '-dir', group_dir
    ]
    
    if RESIZE_IMAGES:
        command.extend(['-resize', 'true'])
    
    # Run evaluation
    try:
        # Run the command and capture output
        result = subprocess.run(
            command, 
            cwd=SCRIPT_DIR, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True,
            check=True
        )
        
        # Save the raw output
        nima_output = result.stdout
        with open(output_file, 'w') as f:
            f.write(nima_output)
            if result.stderr:
                f.write("\n\n# STDERR:\n")
                f.write(result.stderr)
        
        # Parse NIMA output using our custom parser
        nima_image_data, nima_ranking_text = parse_nima_output(nima_output)
        
        if not nima_image_data:
            print(f"Warning: Failed to parse NIMA scores for {category}/{group}")
            group_result['error'] = "Failed to parse NIMA scores"
            group_result['raw_output_file'] = output_file
            results['groups'].append(group_result)
            errors.append(f"{category}/{group}: Failed to parse NIMA scores")
            continue
        
        # Save standardized NIMA output for debugging
        nima_standard_file = os.path.join(RAW_OUTPUT_DIR, f'{category}_{group}_nima_standardized.txt')
        with open(nima_standard_file, 'w') as f:
            # Write image information
            for item in nima_image_data:
                f.write(f"{item['full_text']}\n\n")
            # Write ranking
            f.write(nima_ranking_text)
                
        # Get NIMA ranking
        nima_ranking = get_ranking_from_scores(nima_image_data)
        
        # Extract user preferred data using your script
        user_data, user_ranking_text = extract_text_data(response_path)
        
        if not user_data:
            print(f"Warning: Failed to extract user scores for {category}/{group}")
            group_result['error'] = "Failed to extract user scores"
            group_result['raw_output_file'] = output_file
            group_result['nima_image_data'] = nima_image_data
            results['groups'].append(group_result)
            errors.append(f"{category}/{group}: Failed to extract user scores")
            continue
        
        # Get user ranking
        user_ranking = get_ranking_from_scores(user_data)
        
        # Calculate agreement
        agreement = calculate_rank_agreement(nima_ranking, user_ranking)
        agreements.append(agreement)
        agreement_vector.append(agreement)  # Add to vector for all evaluations
        
        # Check if top ranks match
        top_rank_match = False
        if nima_ranking and user_ranking:
            top_rank_match = nima_ranking[0] == user_ranking[0]
            # Add to top rank vector (1 for match, 0 for no match)
            top_rank_vector.append(1 if top_rank_match else 0)
            if top_rank_match:
                top_rank_matches += 1
        else:
            # If we couldn't determine a match, record as 0
            top_rank_vector.append(0)
        
        # Store results
        group_result.update({
            'success': True,
            'nima_data': nima_image_data,
            'nima_ranking': nima_ranking,
            'user_data': user_data,
            'user_ranking': user_ranking,
            'agreement': agreement,
            'top_rank_match': top_rank_match,
            'raw_output_file': output_file,
            'standard_output_file': nima_standard_file
        })
        
        total_successful += 1
        
    except subprocess.CalledProcessError as e:
        print(f"Error evaluating {category}/{group}: {e}")
        group_result['error'] = f"Process error: {str(e)}"
        errors.append(f"{category}/{group}: Process error - {str(e)}")
    except Exception as e:
        print(f"Error processing {category}/{group}: {e}")
        group_result['error'] = f"Processing error: {str(e)}"
        errors.append(f"{category}/{group}: Processing error - {str(e)}")
    
    results['groups'].append(group_result)

# Calculate summary statistics
if total_successful > 0:
    avg_agreement = np.mean(agreements) if agreements else 0
    top_rank_accuracy = top_rank_matches / total_successful if total_successful > 0 else 0
    
    results['summary'].update({
        'total_groups': len(image_groups),
        'successful_evaluations': total_successful,
        'average_nima_user_agreement': float(avg_agreement),
        'top_rank_accuracy': float(top_rank_accuracy),
        'errors': errors,
        'top_rank_vector': top_rank_vector,
        'agreement_vector': agreement_vector
    })

# Save results
results_file = os.path.join(results_dir, 'nima_evaluation_results.json')
with open(results_file, 'w') as f:
    json.dump(results, f, indent=2)

# Print summary
print("\n===== NIMA EVALUATION SUMMARY =====")
print(f"Total groups in evaluation set: {len(image_groups)}")
print(f"Successful evaluations: {total_successful}")
print(f"Average agreement with user preferred rankings: {avg_agreement:.4f}")
print(f"Top rank accuracy: {top_rank_accuracy:.4f} (how often NIMA's top pick matches user's top pick)")

# Print the vectors
print("\n===== EVALUATION VECTORS =====")
print(f"Top rank match vector [1=match, 0=no match]:")
print(top_rank_vector)
print(f"\nAgreement vector [values between 0-1]:")
print(agreement_vector)

print(f"\nDetailed results saved to: {results_file}")
print(f"Raw outputs saved to: {RAW_OUTPUT_DIR}")
print(f"\nErrors encountered: {len(errors)}")

# Calculate category-specific statistics
if total_successful > 0:
    print("\n===== CATEGORY-SPECIFIC RESULTS =====")
    category_stats = defaultdict(lambda: {
        'count': 0, 
        'agreements': [], 
        'top_matches': 0,
        'top_rank_vector': [],
        'agreement_vector': []
    })
    
    for group_result in results['groups']:
        if group_result['success']:
            category = group_result['category']
            category_stats[category]['count'] += 1
            category_stats[category]['agreements'].append(group_result['agreement'])
            category_stats[category]['agreement_vector'].append(group_result['agreement'])
            top_match = 1 if group_result['top_rank_match'] else 0
            category_stats[category]['top_rank_vector'].append(top_match)
            if group_result['top_rank_match']:
                category_stats[category]['top_matches'] += 1
    
    # Print category stats
    for category, stats in category_stats.items():
        if stats['count'] > 0:
            avg_agreement = np.mean(stats['agreements']) if stats['agreements'] else 0
            top_accuracy = stats['top_matches'] / stats['count'] if stats['count'] > 0 else 0
            print(f"\n{category.upper()}:")
            print(f"  Groups evaluated: {stats['count']}")
            print(f"  Average agreement: {avg_agreement:.4f}")
            print(f"  Top rank accuracy: {top_accuracy:.4f}")
            print(f"  Top rank vector: {stats['top_rank_vector']}")
            print(f"  Agreement vector: {stats['agreement_vector']}")
    
    # Save category stats with vectors
    category_results = {
        category: {
            'count': stats['count'],
            'average_agreement': float(np.mean(stats['agreements'])) if stats['agreements'] else 0,
            'top_rank_accuracy': float(stats['top_matches'] / stats['count']) if stats['count'] > 0 else 0,
            'top_rank_vector': stats['top_rank_vector'],
            'agreement_vector': stats['agreement_vector']
        }
        for category, stats in category_stats.items()
    }
    
    category_file = os.path.join(results_dir, 'nima_category_results.json')
    with open(category_file, 'w') as f:
        json.dump(category_results, f, indent=2)