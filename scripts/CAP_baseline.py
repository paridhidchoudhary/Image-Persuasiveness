import os
import torch
import json
import re
import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText, AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
import scipy.spatial.distance as distance
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# All helper functions remain the same as in the previous version
def load_vision_language_model():
    """Load vision-language model for image description."""
    print("  Loading vision-language model...")
    processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = AutoModelForImageTextToText.from_pretrained("Salesforce/blip-image-captioning-large")
    return model, processor

def load_language_model():
    """Load language model for action-reason generation and persuasiveness scoring."""
    print("  Loading language model...")
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")
    return model, tokenizer

def load_clip_model():
    """Load CLIP model for image-text similarity."""
    print("  Loading CLIP model...")
    from transformers import CLIPProcessor, CLIPModel
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor

def load_similarity_model():
    """Load model for semantic similarity calculation."""
    print("  Loading similarity model...")
    return SentenceTransformer('all-MiniLM-L6-v2')


def generate_image_description(img_path, vl_model, vl_processor):
    """Generate a detailed description for an image."""
    try:
        # Verify image path
        if not os.path.exists(img_path):
            print(f"Image not found: {img_path}")
            return "No description available"
        
        # Load image
        image = Image.open(img_path).convert("RGB")
        
        # Validate image
        if image.width == 0 or image.height == 0:
            print(f"Invalid image: {img_path}")
            return "No description available"
        
        prompt = "Describe this product image in detail for marketing purposes:"
        
        # More explicit inputs processing
        inputs = vl_processor(
            text=prompt, 
            images=image, 
            return_tensors="pt", 
            padding=True,
            truncation=True
        )
        
        with torch.no_grad():
            outputs = vl_model.generate(**inputs, max_new_tokens=150)
        
        description = vl_processor.decode(outputs[0], skip_special_tokens=True)
        return description
    
    except Exception as e:
        print(f"Error processing image {img_path}: {e}")
        return "No description available"

def generate_action_reason(description, llm_model, llm_tokenizer):
    """Generate an action-reason statement from an image description."""
    prompt = f"""
    What is the correct interpretation for the described image:
    Description: {description}
    
    The interpretation format is: I should [action] because [reason]. ONLY RETURN A SINGLE SENTENCE IN THIS FORMAT
    """
    
    inputs = llm_tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    with torch.no_grad():
        outputs = llm_model.generate(**inputs, max_length=150)
    
    generated_ar = llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Ensure output is in correct format
    if "I should" not in generated_ar or "because" not in generated_ar:
        generated_ar = f"I should buy this product because {generated_ar}"
    
    return generated_ar

def calculate_cite_score(generated_ar, expected_ar, similarity_model):
    """Calculate CITE score (alignment) between generated and expected action-reason statements."""
    # Extract action and reason components
    gen_action = re.search(r'I should (.*?) because', generated_ar)
    gen_reason = re.search(r'because (.*?)(?:\.|\Z)', generated_ar)
    
    exp_action = re.search(r'I should (.*?) because', expected_ar)
    exp_reason = re.search(r'because (.*?)(?:\.|\Z)', expected_ar)
    
    # Handle missing components
    if not gen_action or not gen_reason:
        return 0.3  # Default low score
    
    if not exp_action or not exp_reason:
        return 0.3  # Default low score
    
    gen_action, gen_reason = gen_action.group(1), gen_reason.group(1)
    exp_action, exp_reason = exp_action.group(1), exp_reason.group(1)
    
    # Encode sentences for similarity calculation
    gen_action_emb = similarity_model.encode(gen_action, convert_to_tensor=True)
    gen_reason_emb = similarity_model.encode(gen_reason, convert_to_tensor=True)
    
    exp_action_emb = similarity_model.encode(exp_action, convert_to_tensor=True)
    exp_reason_emb = similarity_model.encode(exp_reason, convert_to_tensor=True)
    
    # Calculate cosine similarities
    action_sim = torch.nn.functional.cosine_similarity(gen_action_emb, exp_action_emb, dim=0).item()
    reason_sim = torch.nn.functional.cosine_similarity(gen_reason_emb, exp_reason_emb, dim=0).item()
    
    # Weight reason similarity 4x as in the paper
    alpha = 4
    cite_score = (action_sim + alpha * reason_sim) / (1 + alpha)
    
    return cite_score

def extract_objects_from_text(text):
    """Extract objects mentioned in text for creativity calculation."""
    # Simple extraction based on POS tagging
    words = text.lower().split()
    
    # Filter common words and keep likely objects
    stopwords = ["i", "should", "the", "a", "an", "it", "is", "are", "and", "because", "this", "that", "high", "quality"]
    objects = [word for word in words if word not in stopwords and len(word) > 3]
    
    # Add the category explicitly
    if "buy" in text.lower():
        category_match = re.search(r'buy\s+this\s+(\w+)', text.lower())
        if category_match:
            objects.append(category_match.group(1))
    
    return list(set(objects))  # Remove duplicates

def calculate_creativity_score(img_path, cite_score, objects, clip_model, clip_processor):
    """Calculate creativity score (Cobj) as defined in the CAP paper."""
    # Load image
    image = Image.open(img_path).convert("RGB")
    image_inputs = clip_processor(images=image, return_tensors="pt")
    
    # Get image features
    with torch.no_grad():
        image_features = clip_model.get_image_features(**image_inputs)
    
    # Calculate similarity for each object
    similarities = []
    for obj in objects:
        text_inputs = clip_processor(text=[f"A {obj}"], return_tensors="pt")
        with torch.no_grad():
            text_features = clip_model.get_text_features(**text_inputs)
        
        similarity = torch.nn.functional.cosine_similarity(image_features, text_features).item()
        similarities.append(similarity)
    
    # Calculate average similarity
    avg_similarity = np.mean(similarities) if similarities else 0.5
    
    # Creativity is CITE score divided by object similarity (higher when image shows more than just objects)
    # Add small constant to prevent division by zero
    creativity_score = cite_score / (avg_similarity + 0.1)
    
    # Normalize to 0-1 range
    return min(max(creativity_score * 0.5, 0), 1)  # Scale to ensure normal range

def score_persuasion_component(description, component, category, llm_model, llm_tokenizer):
    """Score an image on a specific persuasion component."""
    component_prompts = {
        "elaboration": f"""
            Context: You are scoring an image based on its description.
            Description: {description}
            
            Question: How visually detailed is the image? Do not consider text in the image.
            Score in range (0, 5) where 5 is extremely detailed.
            
            Your output format should be: Answer: [score]
        """,
        
        "synthesis": f"""
            Context: You are scoring an image based on its description.
            Description: {description}
            
            Question: How well does the image connect elements that are usually unrelated?
            Score in range (0, 5) where 5 means excellent connection of usually unrelated elements.
            
            Your output format should be: Answer: [score]
        """,
        
        "originality": f"""
            Context: You are scoring an image based on its description.
            Description: {description}
            
            Question: How out of the ordinary and unique is the image? How well does it break away 
            from habit-bound and stereotypical thinking?
            Score in range (0, 5) where 5 is extremely original.
            
            Your output format should be: Answer: [score]
        """,
        
        "imagination": f"""
            Context: You are scoring an image based on its description.
            Description: {description}
            
            Question: How well does the image help the audience imagine something they have not
            directly experienced before?
            Score in range (0, 5) where 5 means excellent at stimulating imagination.
            
            Your output format should be: Answer: [score]
        """,
        
        "audience": f"""
            Context: You are scoring an image based on its description.
            Description: {description}
            
            Question: How well does the image target the appropriate audience for a {category} product?
            Score in range (0, 5) where 5 means perfectly targeted to the right audience.
            
            Your output format should be: Answer: [score]
        """,
        
        "benefit": f"""
            Context: You are scoring an image based on its description.
            Description: {description}
            
            Question: How well does the image convert the features of the {category} into benefits for customers?
            Score in range (0, 5) where 5 means excellent conversion of features to benefits.
            
            Your output format should be: Answer: [score]
        """,
        
        "appeal": f"""
            Context: You are scoring an image based on its description.
            Description: {description}
            
            There are three types of rhetorical appeals:
            - Ethos: Appeals to credibility or authority
            - Pathos: Appeals to emotion
            - Logos: Appeals to logic and reason
            
            Question: How well does the image use appropriate appeal techniques (ethos, pathos, logos)
            for a {category} product?
            Score in range (0, 5) where 5 means excellent use of appropriate appeals.
            
            Your output format should be: Answer: [score]
        """
    }
    
    # Get prompt for the component
    prompt = component_prompts.get(component, "")
    if not prompt:
        return 0.5  # Default middle score
    
    # Generate response from LLM
    inputs = llm_tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    with torch.no_grad():
        outputs = llm_model.generate(**inputs, max_length=100)
    
    response = llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract score from response
    score_match = re.search(r'Answer:\s*(\d+(\.\d+)?)', response)
    
    if score_match:
        score = float(score_match.group(1))
        return score / 5.0  # Normalize to 0-1 range
    else:
        # Fallback regex patterns
        number_match = re.search(r'(\d+(\.\d+)?)\s*/\s*5', response) or re.search(r'(\d+(\.\d+)?)', response)
        if number_match:
            score = float(number_match.group(1))
            return min(score / 5.0, 1.0)  # Normalize to 0-1 range with cap
        
        return 0.5  # Default middle score


def calculate_ranking_agreement(cap_ranking, model_ranking):
    """Calculate agreement between CAP ranking and model ranking."""
    if not model_ranking or not cap_ranking:
        return 0.0
    
    # Find common images
    common_images = set(cap_ranking) & set(model_ranking)
    
    if not common_images:
        return 0.0
    
    # Filter rankings to only include common images
    filtered_cap = [img for img in cap_ranking if img in common_images]
    filtered_model = [img for img in model_ranking if img in common_images]
    
    # Find the minimum length of both lists
    min_len = min(len(filtered_cap), len(filtered_model))
    
    # Calculate how many positions have the same image
    matches = sum(1 for i in range(min_len) if filtered_cap[i] == filtered_model[i])
    
    # Return fraction of matching positions
    return matches / min_len if min_len > 0 else 0.0

def parse_model_output(output_file, images=None):
    """Parse model output file to extract image rankings."""
    from simple_data_preprocess import extract_text_data
    
    try:
        # Extract text data from the file
        extracted_info, _ = extract_text_data(output_file)
        
        if not extracted_info:
            return []
        
        # Sort extracted info by scores in descending order
        sorted_info = sorted(extracted_info, key=lambda x: x.get('score', 0), reverse=True)
        
        # Extract image names in order of ranking
        # If no images list is provided, extract from the sorted info
        if images is None:
            return [f"edit_{item['image_num']}_{os.path.basename(output_file).split('_')[1]}.jpg" 
                    for item in sorted_info]
        
        # If images list is provided, match rankings to actual image filenames
        ranked_images = []
        for item in sorted_info:
            # Try to find a matching image
            matching_images = [
                img for img in images 
                if f"edit_{item['image_num']}" in img or 
                   f"original_{item['image_num']}" in img
            ]
            
            if matching_images:
                ranked_images.append(os.path.basename(matching_images[0]))
        
        return ranked_images
    
    except Exception as e:
        print(f"Error parsing model output: {str(e)}")
        return []

def calculate_position_agreement(cap_ranking, model_ranking):
    """Calculate position-wise agreement between CAP ranking and model ranking."""
    if not cap_ranking or not model_ranking:
        return []
    
    # Find common images
    common_images = set(cap_ranking) & set(model_ranking)
    
    if not common_images:
        return []
    
    # Filter rankings to only include common images
    filtered_cap = [img for img in cap_ranking if img in common_images]
    filtered_model = [img for img in model_ranking if img in common_images]
    
    # Calculate position-wise agreement
    pos_agreement = []
    for i, (cap_img, model_img) in enumerate(zip(filtered_cap, filtered_model)):
        pos_agreement.append(1.0 if cap_img == model_img else 0.0)
    
    return pos_agreement

def calculate_ranking_agreement(cap_ranking, model_ranking):
    """Calculate agreement between CAP ranking and model ranking."""
    if not model_ranking or not cap_ranking:
        return 0.0
    
    # Normalize rankings to ensure fair comparison
    # Create a mapping of images to their rank
    cap_rank_dict = {img: rank for rank, img in enumerate(cap_ranking)}
    model_rank_dict = {img: rank for rank, img in enumerate(model_ranking)}
    
    # Find common images
    common_images = set(cap_ranking) & set(model_ranking)
    
    if not common_images:
        return 0.0
    
    # Calculate rank correlations for common images
    matches = sum(
        1 for img in common_images 
        if cap_rank_dict[img] == model_rank_dict[img]
    )
    
    return matches / len(common_images)

def calculate_overall_statistics(results):
    """Calculate overall statistics for the evaluation."""
    if not results.get("group_results"):
        results["statistics"] = {
            "avg_agreement": 0.0,
            "top_rank_accuracy": 0.0,
            "avg_cite_score": 0.0,
            "avg_creativity_score": 0.0,
            "avg_pa_score": 0.0,
            "component_averages": {}
        }
        return
    
    # Calculate metrics
    agreements = []
    top_rank_matches = []
    
    # Calculate average scores
    all_images = []
    for group in results["group_results"]:
        all_images.extend(group.get("image_evaluations", []))
        
        # Collect agreement scores
        agreement = group.get("agreement_with_model", 0.0)
        agreements.append(agreement)
        
        # Collect top rank match indicators
        top_rank_match = group.get("top_rank_match", 0.0)
        top_rank_matches.append(top_rank_match)
    
    # Average metrics
    avg_agreement = np.mean(agreements) if agreements else 0.0
    top_rank_accuracy = np.mean(top_rank_matches) if top_rank_matches else 0.0
    
    # Average scores for all images
    avg_cite = np.mean([img.get("cite_score", 0) for img in all_images]) if all_images else 0.0
    avg_creativity = np.mean([img.get("creativity_score", 0) for img in all_images]) if all_images else 0.0
    avg_pa = np.mean([img.get("pa_score", 0) for img in all_images]) if all_images else 0.0
    
    # Component averages
    component_avgs = {}
    components = ["elaboration", "synthesis", "originality", "imagination", "audience", "benefit", "appeal"]
    
    for comp in components:
        scores = [img["pa_component_scores"].get(comp, 0) for img in all_images if "pa_component_scores" in img]
        component_avgs[comp] = np.mean(scores) if scores else 0.0
    
    # Store statistics
    results["statistics"] = {
        "avg_agreement": float(avg_agreement),
        "top_rank_accuracy": float(top_rank_accuracy),
        "avg_cite_score": float(avg_cite),
        "avg_creativity_score": float(avg_creativity),
        "avg_pa_score": float(avg_pa),
        "component_averages": {k: float(v) for k, v in component_avgs.items()}
    }

def evaluate_product_images_cap(data_root, output_dir, model_output_dir, test_size=0.05):
    """
    Evaluate product images using the CAP framework, but only on the evaluation set.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load models
    print("Loading models...")
    vl_model, vl_processor = load_vision_language_model()
    llm_model, llm_tokenizer = load_language_model()
    clip_model, clip_processor = load_clip_model()
    similarity_model = load_similarity_model()
    
    # Load the dataset following the same approach as your code
    print("Loading dataset...")
    all_data = []
    MAX_IMAGES = 4  # Maximum images per group
    
    for category in os.listdir(data_root):
        category_path = os.path.join(data_root, category)
        if os.path.isdir(category_path):
            for group in os.listdir(category_path):
                group_path = os.path.join(data_root, category, group)
                if os.path.isdir(group_path):
                    images = sorted([
                        os.path.join(group_path, img)
                        for img in os.listdir(group_path)
                        if img.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
                    ])
                    
                    if len(images) > MAX_IMAGES or len(images) == 0:
                        continue
                    
                    # Check that ground truth file exists
                    user_preferred_path = os.path.join(model_output_dir, category, group, "user_output.txt")
                    
                    if os.path.exists(user_preferred_path):
                        # Include this group
                        all_data.append({
                            "images": images,
                            "category": category,
                            "group": group,
                            "ground_truth_path": user_preferred_path
                        })
    
    print(f"Dataset loaded with {len(all_data)} groups having ground truth")
    
    # Split the dataset using the same parameters as in your code
    train_data, test_data = train_test_split(all_data, test_size=test_size, random_state=48)
    
    print(f"Evaluation set size: {len(test_data)} groups")
    
    results = {
        "total_groups": len(test_data),
        "successful_evaluations": 0,
        "errors": 0,
        "group_results": []
    }
    
    # Process only the validation set
    for idx, batch in enumerate(test_data):
        category = batch["category"]
        group = batch["group"]
        images = batch["images"]
        ground_truth_path = batch["ground_truth_path"]
        
        print(f"Processing {idx+1}/{len(test_data)}: {category}/{group}")
        
        try:
            # Parse model's ranking from output
            model_ranking = parse_model_output(ground_truth_path, images)
            
            # For this category, create an action-reason statement
            expected_ar = f"I should buy this {category} because it is high quality and effective"
            
            # Extract objects for creativity calculation
            objects = extract_objects_from_text(expected_ar)
            
            # Evaluate each image
            image_evaluations = []
            for img_path in images:
                img_name = os.path.basename(img_path)
                print(f"  Evaluating {img_name}...")
                
                # Generate image description
                description = generate_image_description(img_path, vl_model, vl_processor)
                
                # Generate action-reason statement
                generated_ar = generate_action_reason(description, llm_model, llm_tokenizer)
                
                # Calculate CITE score (alignment)
                cite_score = calculate_cite_score(generated_ar, expected_ar, similarity_model)
                
                # Calculate creativity score
                creativity_score = calculate_creativity_score(
                    img_path, cite_score, objects, clip_model, clip_processor
                )
                
                # Calculate persuasiveness scores for each component
                pa_component_scores = {}
                for component in ["elaboration", "synthesis", "originality", 
                                 "imagination", "audience", "benefit", "appeal"]:
                    pa_component_scores[component] = score_persuasion_component(
                        description, component, category, llm_model, llm_tokenizer
                    )
                
                # Calculate overall persuasiveness (PA score)
                all_scores = list(pa_component_scores.values()) + [cite_score]
                pa_score = sum(all_scores) / len(all_scores)
                
                # Calculate overall CAP score
                cap_score = (cite_score + creativity_score + pa_score) / 3
                
                # Store evaluation results
                image_evaluations.append({
                    "image": img_name,
                    "description": description,
                    "generated_ar": generated_ar,
                    "cite_score": float(cite_score),
                    "creativity_score": float(creativity_score),
                    "pa_component_scores": {k: float(v) for k, v in pa_component_scores.items()},
                    "pa_score": float(pa_score),
                    "cap_score": float(cap_score)
                })
            
            # Rank images by persuasiveness (PA) score
            ranked_images = sorted(
                image_evaluations, 
                key=lambda x: x["pa_score"], 
                reverse=True
            )

            # Get the rankings as lists of image names
            cap_ranking = [r["image"] for r in ranked_images]
            model_ranking = parse_model_output(ground_truth_path, images)

            # Calculate agreement with model ranking
            agreement = calculate_ranking_agreement(cap_ranking, model_ranking)

            # Calculate top rank match (1 if top ranks match, 0 otherwise)
            top_rank_match = 1.0 if (cap_ranking and model_ranking and cap_ranking[0] == model_ranking[0]) else 0.0

            # Store group result
            group_result = {
                "category": category,
                "group": group,
                "images": images,
                "ground_truth_path": ground_truth_path,
                "top_image_cap": ranked_images[0]["image"],
                "cap_ranking": cap_ranking,
                "model_ranking": model_ranking,
                "agreement_with_model": agreement,
                "top_rank_match": top_rank_match,  # Binary indicator of top rank match
                "image_evaluations": ranked_images
            }
            
            results["group_results"].append(group_result)
            results["successful_evaluations"] += 1
            
            # Save individual group result
            with open(os.path.join(output_dir, f"{category}_{group}_cap_eval.json"), "w") as f:
                json.dump(group_result, f, indent=2)
            
            print(f"Top image: {ranked_images[0]['image']} (PA Score: {ranked_images[0]['pa_score']:.4f})")
            
        except Exception as e:
            print(f"Error processing {category}/{group}: {str(e)}")
            results["errors"] += 1
    
    # Calculate overall statistics
    calculate_overall_statistics(results)
    
    # Save overall results
    with open(os.path.join(output_dir, "cap_evaluation_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    return results


def main():
    ## SET PATHS
    data_root = "/home/deepg/NAS/Downloads/MTP-2-persuasion/dataset/dataset_image"
    model_output_dir = "/home/deepg/NAS/Downloads/MTP-2-persuasion/dataset/final_data" 
    output_dir = "cap_evaluation_results"
    test_size = 0.05  # Same as your code
    
    print("Starting CAP evaluation of product images on evaluation set...")
    results = evaluate_product_images_cap(data_root, output_dir, model_output_dir, test_size)
    
    print("\nEvaluation Complete!")
    print(f"Total groups in evaluation set: {results['total_groups']}")
    print(f"Successfully evaluated: {results['successful_evaluations']}")
    print(f"Errors: {results['errors']}")
    
    if "statistics" in results:
        print("\nOverall Statistics:")
        print(f"Average agreement with ground truth: {results['statistics']['avg_agreement']:.4f}")
        print(f"Top rank accuracy: {results['statistics']['top_rank_accuracy']:.4f}")
        print(f"Average CITE score: {results['statistics']['avg_cite_score']:.4f}")
        print(f"Average creativity score: {results['statistics']['avg_creativity_score']:.4f}")
        print(f"Average PA score: {results['statistics']['avg_pa_score']:.4f}")
        
        # Extract and print the vectors
        top_rank_matches = [group.get("top_rank_match", 0) for group in results["group_results"]]
        agreements = [group.get("agreement_with_model", 0) for group in results["group_results"]]
        
        print("\nDetailed Metrics Vectors:")
        print(f"Top Rank Match Vector: {top_rank_matches}")
        print(f"Agreement Vector: {agreements}")
        
        print("\nPersuasiveness Component Averages:")
        for comp, avg in results['statistics']['component_averages'].items():
            print(f"  {comp}: {avg:.4f}")
    
    # Also save these vectors to separate files for easier analysis
    if results.get("group_results"):
        top_rank_matches = [group.get("top_rank_match", 0) for group in results["group_results"]]
        agreements = [group.get("agreement_with_model", 0) for group in results["group_results"]]
        
        # Save as JSON
        with open(os.path.join(output_dir, "top_rank_matches_vector.json"), "w") as f:
            json.dump(top_rank_matches, f)
            
        with open(os.path.join(output_dir, "agreements_vector.json"), "w") as f:
            json.dump(agreements, f)
        
        # Also save as plain text for easy copy-paste
        with open(os.path.join(output_dir, "top_rank_matches_vector.txt"), "w") as f:
            f.write(str(top_rank_matches))
            
        with open(os.path.join(output_dir, "agreements_vector.txt"), "w") as f:
            f.write(str(agreements))
    
    print(f"\nResults saved to: {output_dir}")
    print(f"Summary file: {os.path.join(output_dir, 'cap_evaluation_results.json')}")
    print(f"Vectors saved as: top_rank_matches_vector.json, agreements_vector.json")


if __name__ == "__main__":
    main()