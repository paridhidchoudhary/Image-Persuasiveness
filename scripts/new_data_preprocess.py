import os
import re
from PIL import Image

# ---------------------------------------------------------------------------------
# Update these paths as needed:
# ---------------------------------------------------------------------------------
## SET PATHS
data_root = "home/debajyoti/paridhi_mtp/product_images_real"
#data_root = "/home/krudra/debajyoti"
dataset_image = os.path.join(data_root, "dataset_image")
#dataset_image = os.path.join(data_root, "dataset/dataset_image")
dataset_response = os.path.join(data_root, "final_data")
# Updated path to user preferred outputs (ground truth)
dataset_user_preferred = os.path.join(data_root, "final_data")
MAX_IMAGES = 4




def extract_bullet_format_scores(text):
    """
    Extract scores from a bullet point format like:
    - **Persuasion Score:** 70
    """
    bullet_pattern = r'-\s+\*\*Persuasion Score:\*\*\s+(\d+)'
    matches = re.finditer(bullet_pattern, text)
    scores = []
    
    for match in matches:
        try:
            score = int(match.group(1))
            scores.append(score)
        except (ValueError, IndexError):
            continue
            
    return scores

def process_image_section_with_bullets(raw_text):
    """
    Process text formatted with '#### **Image X**' headers and bullet points
    Returns list of image data dicts with image_num, score, and full_text
    """
    # Pattern to match image header and content until next header or section
    image_pattern = r'####\s+\*\*Image\s+(\d+)\*\*([\s\S]*?)(?=####\s+\*\*Image|\n*###\s+|$)'
    matches = re.finditer(image_pattern, raw_text)
    
    image_data = []
    for match in matches:
        try:
            image_num = int(match.group(1))
            section_text = match.group(0)
            
            # Extract score from bullet format
            score_pattern = r'-\s+\*\*Persuasion Score:\*\*\s+(\d+)'
            score_match = re.search(score_pattern, section_text)
            
            if score_match:
                score = int(score_match.group(1))
                image_data.append({
                    'image_num': image_num,
                    'full_text': section_text,
                    'score': score
                })
            else:
                # Try other score extraction methods
                score = extract_score(section_text)
                if score is not None:
                    image_data.append({
                        'image_num': image_num,
                        'full_text': section_text,
                        'score': score
                    })
        except (ValueError, IndexError) as e:
            print(f"Error processing image section: {e}")
            continue
            
    return image_data

def extract_from_ranking_section(ranking_text, missing_image_nums):
    """
    Extract scores from a ranking section for specific image numbers
    """
    results = []
    
    # Try different patterns for matching image scores in ranking text
    patterns = [
        # 1. **Image 2** - text text 75 text
        r'(\d+)\.\s+\*\*Image\s+(\d+)\*\*.*?(\d+)',
        # 1. **Image 2:** text Score: 75 text
        r'(\d+)\.\s+\*\*Image\s+(\d+):\*\*.*?Score:\s*(\d+)',
        # **Image 2** ranks higher with score 75
        r'\*\*Image\s+(\d+)\*\*.*?score\s+(\d+)',
        # Image 2 - Persuasion Score: 75
        r'Image\s+(\d+).*?Persuasion\s+Score:\s*(\d+)'
    ]
    
    for pattern in patterns:
        matches = re.finditer(pattern, ranking_text, re.IGNORECASE | re.DOTALL)
        for match in matches:
            try:
                # The image number is in group 1 or 2 depending on the pattern
                if pattern.startswith(r'(\d+)'):
                    image_id = int(match.group(2))
                    score = int(match.group(3))
                else:
                    image_id = int(match.group(1))
                    score = int(match.group(2))
                
                if image_id in missing_image_nums:
                    results.append({
                        'image_num': image_id,
                        'score': score,
                        'full_text': ''  # We don't have the full text here
                    })
            except (ValueError, IndexError):
                continue
                
    return results

def find_ranking_position(text):
    """Find the starting position of the ranking section"""
    # Specific pattern for the new format in the example
    evaluation_header = r'###\s+Evaluation\s+of\s+New\s+Images'
    eval_match = re.search(evaluation_header, text, re.MULTILINE)
    if eval_match:
        # For this format, check if there's a "### Ranking" section after the evaluation
        ranking_header = r'###\s+Ranking'
        rank_match = re.search(ranking_header, text[eval_match.end():], re.MULTILINE)
        if rank_match:
            return eval_match.end() + rank_match.start()
    
    # First check for ranking headers and persuasion scores specific to the new format
    standalone_ranking = r'###\s+Ranking\s*$'
    standalone_match = re.search(standalone_ranking, text, re.MULTILINE)
    if standalone_match:
        return standalone_match.start()
        
    ranking_persuasion_pattern = r'###\s+Ranking\s+and\s+Persuasion\s+Scores\s+for\s+Images'
    ranking_match = re.search(ranking_persuasion_pattern, text, re.IGNORECASE)
    if ranking_match:
        return ranking_match.start()
    
    # Check for ranking and explanation pattern
    ranking_explanation_pattern = r'###\s+Ranking\s+and\s+Explanation:'
    expl_match = re.search(ranking_explanation_pattern, text, re.IGNORECASE)
    if expl_match:
        return expl_match.start()
    
    # Below this is the existing pattern list - continue with all the patterns that were working
    ranking_patterns = [
        r'^(?:###\s+Ranking(?:\s+Explanation)?:)',
        r'^(?:Ranking(?:\s+Summary)?:)',
        r'^(?:###\s+Ranking(?:\s+Explanation)?$)',
        r'^(?:Ranking(?:\s+Explanation)?:)',
        r'^(?:###\s+Ranking:$)',
        r'^(?:###\s+Ranking$)',
        r'^(?:###\s+Summary$)',
        r'^(?:Summary:$)',
        r'^(?:###\s+Explanation of Ranking:$)',
        r'^(?:Explanation of Ranking:$)',
        r'^(?:###\s+Ranking\b)',
        r'^(?:###\s+Ranking\s+Explanation:)',
        r'^(?:###\s+Ranking\s+Summary:)',
        r'^(?:###\s+Ranking\s+Explanation)',
        r'^(?:###\s+Overall\s+Ranking)',  # This matches the test case
        r'^(?:###\s+Overall\s+Ranking:)',  # Added with colon
        r'^(?:###\s+Ranking\s+and\s+Persuasion\s+Scores:)',
        r'^(?:Ranking\s+Summary:)',
        r'^(?:RANKING\s*$)',
        r'^(?:\*\*Ranking:\*\*)',
        r'^(?:\d+\.\s+\*\*Image\s+\d+:\*\*\s+Persuasion\s+Score:)',
        r'^(?:\*\*Reasoning:)',
        r'^(?:###\s+Ranking\s+of\s+Images)',  # Added for the test case header
        r'^(?:\*\*Ranking\s+Explanation:\*\*)',  # Added for the specific case described
        r'^(?:Ranking\s+Explanation:)'  # Added for the specific case described
    ]

    for pattern in ranking_patterns:
        ranking_match = re.search(pattern, text, re.MULTILINE)
        if ranking_match:
            return ranking_match.start()
    
    # If we reach here, check for patterns specific to the numbered ranking format
    numbered_ranking_pattern = r'\d+\.\s+\*\*Image\s+\d+:\*\*'
    numbered_match = re.search(numbered_ranking_pattern, text, re.MULTILINE)
    if numbered_match:
        return numbered_match.start()
    
    # If ranking summary wasn't found earlier, look for it now
    ranking_summary_match = re.search(r'\*\*Ranking\s+Summary:\*\*', text)
    if ranking_summary_match:
        return ranking_summary_match.start()
    
    # Also check for standalone RANKING header
    ranking_header_match = re.search(r'^RANKING\s*$', text, re.MULTILINE)
    if ranking_header_match:
        # If found, try to look for a preceding Ranking Explanation section
        text_before = text[:ranking_header_match.start()]
        ranking_expl_match = re.search(r'\*\*Ranking\s+Explanation:\*\*(.*?)$', text_before, re.DOTALL)
        if ranking_expl_match:
            return ranking_expl_match.start()
        else:
            return ranking_header_match.start()
    
    # Look for numbered list with image scores (common ranking format)
    ranked_list_patterns = [
        r'^\d+\.\s+Image\s+\d+\s+\(Score:\s+\d+\)',
        r'^\d+\.\s+\*\*Image\s+\d+:\*\*\s+\d+/100',  # Added for formats like "1. **Image 1:** 85/100"
    ]
    
    for pattern in ranked_list_patterns:
        ranked_list_match = re.search(pattern, text, re.MULTILINE)
        if ranked_list_match:
            return ranked_list_match.start()
    
    # Add check for "#### Image X:" patterns followed by Persuasion Score
    image_score_pattern = r'####\s+Image\s+\d+:\s*\n-\s+\*\*Persuasion\s+Score\*\*:'
    image_score_match = re.search(image_score_pattern, text, re.MULTILINE)
    if image_score_match:
        return image_score_match.start()
    
    # Add check for patterns like "1. **Image 1:** 85/100" in the overall ranking
    rank_pattern = r'\d+\.\s+\*\*Image\s+\d+:\*\*\s+\d+/100'
    rank_match = re.search(rank_pattern, text, re.MULTILINE)
    if rank_match:
        return rank_match.start()
        
    # Look for conclusion section as a fallback
    conclusion_pattern = r'###\s+Conclusion'
    conclusion_match = re.search(conclusion_pattern, text, re.MULTILINE)
    if conclusion_match:
        return conclusion_match.start()

    return len(text)  # Default to end of text if no ranking section found


def extract_score(text):
    """Extract the persuasion score from text using various patterns"""
    # First check for the specific format in the examples with actual 2-digit numbers
    specific_patterns = [
        r'\*\*Persuasion Score: (\d{1,3})\*\*',            # **Persuasion Score: 85**
        r'Persuasion Score: (\d{1,3})(?![0-9/])',          # Persuasion Score: 85
        r'\*\*Persuasion Score:\*\* (\d{1,3})',            # **Persuasion Score:** 85
    ]
    
    for pattern in specific_patterns:
        score_match = re.search(pattern, text)
        if score_match:
            try:
                score = int(score_match.group(1))
                if 1 <= score <= 100:  # Validate range
                    return score
            except ValueError:
                pass
    
    # The rest of your patterns - NOTE: changed (\d+) to (\d{1,3}) to ensure we get full numbers
    score_patterns = [
        r'-\s+\*\*Persuasion Score:\*\*\s+(\d{1,3})(?!\s*/)',  # - **Persuasion Score:** 70
        r'-\s+\*\*Persuasion Score:\*\*\s+(\d{1,3})/100',      # - **Persuasion Score:** 70/100
        r'-\s+\*\*Persuasion Score:\*\*\s+(\d{1,3})',          # - **Persuasion Score:** 70
        r'Persuasion Score is (\d{1,3})',                      # Persuasion Score is 70
        r'Persuasion Score is \*\*(\d{1,3})\*\*',              # Persuasion Score is **70**
        r'Persuasion Score is \*\*(\d{1,3})/100\*\*',          # Persuasion Score is **70/100**
        r'Persuasion Score is (\d{1,3})/100',                  # Persuasion Score is 70/100
        r'persuasion score is (\d{1,3})',                      # persuasion score is 70
        r'persuasion score is \*\*(\d{1,3})\*\*',              # persuasion score is **70**
        r'persuasion score is \*\*(\d{1,3})/100\*\*',          # persuasion score is **70/100**
        r'persuasion score is (\d{1,3})/100',                  # persuasion score is 70/100
        r'\*\*Persuasion Score:\*\*\s*\*\*(\d{1,3})/100\*\*',
        r'\*\*Persuasion Score:\*\*\s*(\d{1,3})/100',
        r'\*\*Persuasion Score\*\*:\s*(\d{1,3})/100',
        r'Persuasion Score:\s*(\d{1,3})/100',
        r'-\s+\*\*Persuasion Score\*\*:\s*(\d{1,3})/100',
        r'-\s+Persuasion Score:\s*(\d{1,3})/100',
        r'\*\*Persuasion Score:\*\*\s*(\d{1,3})',
        r'\*\*Persuasion Score\*\*:\s*(\d{1,3})',
        r'Persuasion Score:\s*(\d{1,3})',
        r'-\s+\*\*Persuasion Score\*\*:\s*(\d{1,3})',
        r'-\s+Persuasion Score:\s*(\d{1,3})',
        r'-\s+\*\*Persuasion Score:\*\*\s*(\d{1,3})',
        r'\*\*Persuasion Score:\*\*\s*\*\*(\d{1,3})/100\*\*',
        r'-\s+\*\*Persuasion Score:\*\*\s+(\d{1,3})(?!\s*/)',
        r'Persuasion Score:\*\*\s+(\d{1,3})(?!\s*/)',
        r'\*\*Persuasion Score:\*\*\s+(\d{1,3})(?!\s*/)',
        r'\(Persuasion Score: (\d{1,3})\)',
        r'\*\*Persuasion Score\*\*:\s+(\d{1,3})/100',
        r'\*\*Persuasion Score:\*\*\s+\*\*(\d{1,3})',
        r'\*\*Persuasion Score:\*\*\s+\*\*(\d{1,3})/100',
        r'Persuasion Score:\*\*\s*\*\*(\d{1,3})/100',
        r'Persuasion Score:\*\*\s*\*\*(\d{1,3})',
        r'Persuasion Score:\s*\*\*(\d{1,3})/100\*\*',
        r'Persuasion Score:\s*\*\*(\d{1,3})\*\*',
        r'Persuasion Score:\s*(\d{1,3})/100',
        r'Persuasion Score\*\*:\s*(\d{1,3})/100',
        r'Persuasion Score: (\d{1,3})/100',        
        r'\*\*Persuasion Score: (\d{1,3})/100\*\*',
        r'\*\*Persuasion Score: (\d{1,3})\*\*',
        r'Persuasion Score: (\d{1,3})',        
        r'Persuasion\s+Score.*?(\d{1,3})',
    ]
    
    # Try each pattern
    score_match = re.search(pattern, text)
    if score_match:
        raw_score = score_match.group(1)
        print(f"Matched pattern '{pattern}' with raw score: '{raw_score}'")
        try:
            score = int(raw_score)
            if 1 <= score <= 100:
                return score
        except ValueError:
            pass
    
    return None

def extract_score(text):
    """Extract the persuasion score from text using various patterns"""
    # Handle most common format first - standard X/100 patterns
    standard_patterns = [
        r'\*\*Persuasion Score:\*\*\s*(\d+)/100',      # **Persuasion Score:** 70/100
        r'\*\*Persuasion Score\*\*:\s*(\d+)/100',      # **Persuasion Score**: 70/100
        r'Persuasion Score:\s*(\d+)/100',              # Persuasion Score: 70/100
        r'-\s+\*\*Persuasion Score\*\*:\s*(\d+)/100',  # - **Persuasion Score**: 70/100
        r'-\s+\*\*Persuasion Score:\*\*\s*(\d+)/100',  # - **Persuasion Score:** 70/100
        r'-\s+Persuasion Score:\s*(\d+)/100',          # - Persuasion Score: 70/100
    ]
    
    for pattern in standard_patterns:
        match = re.search(pattern, text, re.MULTILINE)
        if match:
            try:
                score = int(match.group(1))
                if 1 <= score <= 100:  # Validate range
                    return score
            except ValueError:
                pass
    
    # Handle cases with bold formatting
    bold_patterns = [
        r'\*\*Persuasion Score:\*\*\s*\*\*(\d+)/100\*\*',  # **Persuasion Score:** **70/100**
        r'\*\*Persuasion Score:\*\*\s*\*\*(\d+)\*\*/100',  # **Persuasion Score:** **70**/100
        r'Persuasion Score:\s*\*\*(\d+)/100\*\*',          # Persuasion Score: **70/100**
        r'Persuasion Score:\s*\*\*(\d+)\*\*/100',          # Persuasion Score: **70**/100
        r'\*\*Persuasion Score:\*\*\s*\*\*(\d+)\*\*',      # **Persuasion Score:** **70**
        r'Persuasion Score:\s*\*\*(\d+)\*\*',              # Persuasion Score: **70**
    ]
    
    for pattern in bold_patterns:
        match = re.search(pattern, text, re.MULTILINE)
        if match:
            try:
                score = int(match.group(1))
                if 1 <= score <= 100:
                    return score
            except ValueError:
                pass
    
    # Handle the specific format in your examples (no /100 suffix)
    no_suffix_patterns = [
        r'\*\*Persuasion Score:\*\*\s*(\d{1,3})(?!\d|/)',  # **Persuasion Score:** 70 (not followed by digit or /)
        r'\*\*Persuasion Score\*\*:\s*(\d{1,3})(?!\d|/)',  # **Persuasion Score**: 70
        r'Persuasion Score:\s*(\d{1,3})(?!\d|/)',         # Persuasion Score: 70
        r'-\s+\*\*Persuasion Score\*\*:\s*(\d{1,3})(?!\d|/)', # - **Persuasion Score**: 70
        r'-\s+\*\*Persuasion Score:\*\*\s*(\d{1,3})(?!\d|/)', # - **Persuasion Score:** 70
        r'-\s+Persuasion Score:\s*(\d{1,3})(?!\d|/)',     # - Persuasion Score: 70
    ]
    
    for pattern in no_suffix_patterns:
        match = re.search(pattern, text, re.MULTILINE)
        if match:
            try:
                score = int(match.group(1))
                if 1 <= score <= 100:
                    return score
            except ValueError:
                pass
    
    # Try bullet point formats (common in your examples)
    bullet_patterns = [
        r'-\s+\*\*Persuasion Score:\*\*\s+(\d{1,3})',  # - **Persuasion Score:** 70
        r'-\s+Persuasion Score:\s+(\d{1,3})',          # - Persuasion Score: 70
    ]
    
    for pattern in bullet_patterns:
        matches = re.finditer(pattern, text, re.MULTILINE)
        for match in matches:
            try:
                score = int(match.group(1))
                if 1 <= score <= 100:
                    return score
            except ValueError:
                pass
    
    # Try extracting from special formats in ranking sections
    ranking_patterns = [
        r'(\d+)\.\s+\*\*Image\s+\d+\*\*\s+\((\d{1,3})/100\)',  # 1. **Image 1** (70/100)
        r'(\d+)\.\s+\*\*Image\s+\d+:\*\*\s+(\d{1,3})/100',     # 1. **Image 1:** 70/100
        r'(\d+)\.\s+\*\*Image\s+\d+\*\*.*?(\d{2})(?!\d)/100',  # Rank with any text followed by score
        r'(\d+)\.\s+\*\*Image\s+\d+:\*\*.*?(\d{2})(?!\d)/100', 
        r'Image\s+\d+\s+\((\d{1,3})/100\)',                     # Image 1 (70/100)
    ]
    
    for pattern in ranking_patterns:
        matches = re.finditer(pattern, text, re.MULTILINE)
        for match in matches:
            try:
                # The score is in group 2 for most patterns, group 1 for the last
                score_group = 2 if pattern.count(r'\(') > 0 else 1
                score = int(match.group(score_group))
                if 1 <= score <= 100:
                    return score
            except (ValueError, IndexError):
                pass
    
    # Try more general patterns with context validation
    context_patterns = [
        r'score.*?(\d{1,3})/100',         # Any text with "score" followed by digits/100
        r'persuasion.*?(\d{1,3})/100',    # Any text with "persuasion" followed by digits/100
    ]
    
    for pattern in context_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
        for match in matches:
            try:
                score = int(match.group(1))
                if 1 <= score <= 100:
                    # Validate by checking nearby context to avoid false positives
                    context_start = max(0, match.start() - 20)
                    context_end = min(len(text), match.end() + 20)
                    context = text[context_start:context_end].lower()
                    
                    # Only accept if persuasion-related terms are nearby
                    if any(term in context for term in ['persuasion', 'score', 'rating', 'evaluation']):
                        return score
            except ValueError:
                pass
    
    # Last resort - try to find any number between 1-100 in a reasonable context
    number_in_context = re.finditer(r'(?:score|rating|persuasion).*?(\d{1,3})(?!\d)', text, re.IGNORECASE)
    for match in number_in_context:
        try:
            score = int(match.group(1))
            if 1 <= score <= 100:
                return score
        except ValueError:
            pass
    
    return None


def extract_text_data(file_path):
    """
    Enhanced version that extracts image data and rankings from text files.
    Handles additional formats for proper score extraction.
    
    Returns:
        tuple: (image_data, ranking_text)
            - image_data: list of dicts with {'image_num', 'full_text', 'score'}
            - ranking_text: string with the entire ranking explanation
    """
    with open(file_path, "r", encoding="utf-8") as f:
        raw_text = f.read().strip()
    
    print(f"\n[INFO] Processing file: {file_path}")
    
    # Pre-process to normalize formats that cause issues
    normalized_text = normalize_text_format(raw_text)
    if normalized_text != raw_text:
        print("[INFO] Text format normalized for easier processing")
        raw_text = normalized_text
    
    # Check for image headers with bold formatting
    image_header_bold_pattern = r'####\s+\*\*Image\s+(\d+)\*\*([\s\S]*?)(?=####\s+\*\*Image|\n*###\s+|$)'
    image_headers_bold = list(re.finditer(image_header_bold_pattern, raw_text))
    
    # Check for standard image headers
    image_header_pattern = r'####\s+Image\s+(\d+)([\s\S]*?)(?=####\s+Image|\n*###\s+|$)'
    image_headers = list(re.finditer(image_header_pattern, raw_text))
    
    # If either format is found, process sections directly
    if image_headers_bold or image_headers:
        headers_to_use = image_headers_bold if image_headers_bold else image_headers
        print(f"[INFO] Found structured format with {len(headers_to_use)} image headers")
        
        # Find ranking position after all image sections
        last_image_end = headers_to_use[-1].end()
        ranking_section = raw_text[last_image_end:]
        
        # Look for Ranking section
        ranking_header_match = re.search(r'###\s+Ranking', ranking_section)
        if ranking_header_match:
            ranking_position = last_image_end + ranking_header_match.start()
            # Extract the complete ranking text
            ranking_text = extract_ranking(raw_text[ranking_position:])
        else:
            # If no explicit ranking header, use the end of the last image section
            ranking_position = last_image_end
            ranking_text = ""
        
        # Extract data for each image section
        image_data = []
        for match in headers_to_use:
            image_num = int(match.group(1))
            section_text = match.group(0)
            
            # Try to extract score using our improved function
            score = extract_score(section_text)
            
            if score is not None:
                image_data.append({
                    'image_num': image_num,
                    'full_text': section_text.strip(),
                    'score': score
                })
                print(f"[INFO] Extracted score {score} for image {image_num}")
            else:
                print(f"[WARNING] Failed to extract score for image {image_num}")
        
        # Sort image_data by image_num
        image_data.sort(key=lambda x: x['image_num'])
        
        # Verify data integrity
        if not verify_data(image_data):
            print("[WARNING] Data verification failed, results may be incomplete")
            
            # Try to extract missing scores from ranking text
            if ranking_text:
                # Get missing image numbers
                existing_nums = {item['image_num'] for item in image_data}
                max_num = max([m.group(1) for m in headers_to_use], key=int)
                missing_nums = set(range(1, int(max_num) + 1)) - existing_nums
                
                if missing_nums:
                    print(f"[INFO] Attempting to extract scores for missing images: {missing_nums}")
                    from_ranking = extract_from_ranking_section(ranking_text, missing_nums)
                    if from_ranking:
                        image_data.extend(from_ranking)
                        image_data.sort(key=lambda x: x['image_num'])
                        print(f"[INFO] Added {len(from_ranking)} images from ranking section")
        
        # Print extraction stats
        print(f"[INFO] Extracted {len(image_data)} image entries.")
        print(f"[INFO] Images with valid scores: {sum(1 for item in image_data if item['score'] is not None)}")
        print(f"[INFO] Ranking section length: {len(ranking_text)} characters")
        
        return image_data, ranking_text
    
    # If we reach here, use the standard approach for older formats
    # First find the ranking position
    ranking_expl_match = re.search(r'\*\*Ranking\s+Explanation:\*\*(.*?)(?=\n\nRANKING|\Z)', raw_text, re.DOTALL)
    if ranking_expl_match:
        print("[INFO] Found **Ranking Explanation:** section")
        ranking_position = ranking_expl_match.start()
    else:
        # Check for ranking summary pattern 
        ranking_summary_match = re.search(r'(\*\*Ranking\s+Summary:\*\*.*?)(?=\n\n\d+\.\s+\*\*Image|\Z)', raw_text, re.DOTALL)
        
        # Find the position of the ranking section
        ranking_position = find_ranking_position(raw_text)
        
        if ranking_summary_match and ranking_summary_match.start() < ranking_position:
            print("[INFO] Found ranking summary earlier in text, adjusting extraction")
            ranking_position = ranking_summary_match.start()
    
    # Extract the complete ranking text
    ranking_text = extract_ranking(raw_text[ranking_position:])
    
    # Check for Overall Ranking section
    overall_ranking_match = re.search(r'###\s+Overall\s+Ranking', raw_text)
    if overall_ranking_match and overall_ranking_match.start() > ranking_position:
        # Use text before the overall ranking section
        image_text = raw_text[:overall_ranking_match.start()]
    else:
        # Use text before the ranking section
        image_text = raw_text[:ranking_position]
    
    # Identify image headers with various patterns
    header_pattern = r'(?:###\s+Image\s+(\d+)|####\s+Image\s+(\d+)|\*\*Image\s+(\d+):|^\d+\.\s+\*\*Image\s+(\d+)|Image\s+(\d+):|^\d+\.\s+\*\*(First|Second|Third|Fourth|Fifth)\s+Image:|^\d+\.\s+\*\*(First|Second|Third|Fourth|Fifth)\s+Image\b|\*\*(First|Second|Third|Fourth|Fifth)\s+Image:)'
    
    all_headers = list(re.finditer(header_pattern, image_text, re.MULTILINE))
    
    # Process headers to extract image numbers and positions
    sections = []
    for i, match in enumerate(all_headers):
        start_pos = match.start()
        
        # Extract image number from the groups
        image_num = None
        for j in range(1, 6):
            try:
                if match.group(j) and match.group(j).isdigit():
                    image_num = int(match.group(j))
                    break
            except IndexError:
                continue
        
        # Handle textual numbers
        if image_num is None:
            for j in range(6, 9):
                try:
                    if match.group(j):
                        textual_num = match.group(j).lower()
                        if textual_num == "first":
                            image_num = 1
                        elif textual_num == "second":
                            image_num = 2
                        elif textual_num == "third":
                            image_num = 3
                        elif textual_num == "fourth":
                            image_num = 4
                        elif textual_num == "fifth":
                            image_num = 5
                        break
                except IndexError:
                    continue
        
        if image_num is not None:
            # Determine section end
            if i+1 < len(all_headers):
                section_end = all_headers[i+1].start()
            else:
                # For the last section, check for section delimiters
                overall_ranking_pos = raw_text.find("### Overall Ranking", start_pos)
                ranking_of_images_pos = raw_text.find("### Ranking of Images", start_pos)
                
                possible_ends = [pos for pos in [overall_ranking_pos, ranking_of_images_pos, ranking_position] if pos > start_pos]
                section_end = min(possible_ends) if possible_ends else ranking_position
            
            section_text = image_text[start_pos:section_end]
            
            # If this section contains a ranking summary, adjust the end position
            summary_pos = section_text.find("**Ranking Summary:**")
            if summary_pos != -1:
                section_end = start_pos + summary_pos
                print(f"[INFO] Found ranking summary within image section {image_num}, adjusting boundary")
            
            sections.append({
                'image_num': image_num,
                'start': start_pos,
                'end': section_end
            })
    
    # Extract data for each image section
    image_data = []
    processed_image_nums = set()  # Track processed image numbers
    
    for section in sections:
        # Skip if already processed
        if section['image_num'] in processed_image_nums:
            continue
            
        section_text = image_text[section['start']:section['end']].strip()
        
        # Extract score
        score = extract_score(section_text)
        
        if score is not None:
            image_data.append({
                'image_num': section['image_num'],
                'full_text': section_text,
                'score': score
            })
            processed_image_nums.add(section['image_num'])
            print(f"[INFO] Extracted score {score} for image {section['image_num']}")
    
    # Extract missing scores from ranking if needed
    if not image_data or len(image_data) < len(sections):
        missing_image_nums = set(section['image_num'] for section in sections) - processed_image_nums
        print(f"[INFO] Missing scores for images: {missing_image_nums}. Attempting to extract from ranking text.")
        
        # Try to extract from ranking text with various patterns
        if ranking_text:
            missing_scores = extract_from_ranking_section(ranking_text, missing_image_nums)
            if missing_scores:
                image_data.extend(missing_scores)
                for item in missing_scores:
                    processed_image_nums.add(item['image_num'])
                    print(f"[INFO] Extracted score {item['score']} for image {item['image_num']} from ranking section")
        
        # If still not found, try to extract from the specific patterns in overall text
        still_missing = set(section['image_num'] for section in sections) - processed_image_nums
        if still_missing:
            for image_id in still_missing:
                # Find section text
                section_text = ''
                matching_section = next((s for s in sections if s['image_num'] == image_id), None)
                if matching_section:
                    section_text = image_text[matching_section['start']:matching_section['end']].strip()
                    
                    # Try to directly extract with alternate patterns
                    additional_patterns = [
                        r'Persuasion Score: (\d+)/100',
                        r'persuasion score: (\d+)',
                        r'score: (\d+)/100',
                        r'score of (\d+)'
                    ]
                    
                    for pattern in additional_patterns:
                        score_match = re.search(pattern, section_text, re.IGNORECASE)
                        if score_match:
                            try:
                                score = int(score_match.group(1))
                                if 1 <= score <= 100:  # Validate score range
                                    image_data.append({
                                        'image_num': image_id,
                                        'full_text': section_text,
                                        'score': score
                                    })
                                    processed_image_nums.add(image_id)
                                    print(f"[INFO] Extracted score {score} for image {image_id} with alternate pattern")
                                    break
                            except ValueError:
                                continue
    
    # Sort image_data by image_num
    image_data.sort(key=lambda x: x['image_num'])
    
    # Verify data integrity
    if not verify_data(image_data) and sections:
        print("[WARNING] Data verification failed, results may be incomplete")
    
    # Print information about extraction
    print(f"[INFO] Extracted {len(image_data)} image entries.")
    print(f"[INFO] Images with valid scores: {sum(1 for item in image_data if item['score'] is not None)}")
    print(f"[INFO] Ranking section length: {len(ranking_text)} characters")
    
    return image_data, ranking_text


def extract_from_ranking_section(ranking_text, missing_image_nums):
    """
    Extract scores from a ranking section for specific image numbers
    """
    results = []
    
    # Try different patterns for matching image scores in ranking text
    patterns = [
        # 1. **Image 2** - text text 75/100 text
        r'(\d+)\.\s+\*\*Image\s+(\d+)\*\*.*?(\d+)/100',
        # 1. **Image 2:** text Score: 75/100 text
        r'(\d+)\.\s+\*\*Image\s+(\d+):\*\*.*?Score:\s*(\d+)/100',
        # 1. **Image 2:** 75/100
        r'(\d+)\.\s+\*\*Image\s+(\d+):\*\*\s+(\d+)/100',
        # **Image 2** ranks higher with score 75/100
        r'\*\*Image\s+(\d+)\*\*.*?score\s+(\d+)/100',
        # Image 2 - Persuasion Score: 75/100
        r'Image\s+(\d+).*?Persuasion\s+Score:\s*(\d+)/100',
        # Generic digit pattern near Image X
        r'Image\s+(\d+).*?(\d{1,3})/100'
    ]
    
    for pattern in patterns:
        matches = re.finditer(pattern, ranking_text, re.IGNORECASE | re.DOTALL)
        for match in matches:
            try:
                # The image number is in group 1 or 2 depending on the pattern
                if pattern.startswith(r'(\d+)'):
                    image_id = int(match.group(2))
                    score = int(match.group(3))
                else:
                    image_id = int(match.group(1))
                    score = int(match.group(2))
                
                if image_id in missing_image_nums and 1 <= score <= 100:
                    results.append({
                        'image_num': image_id,
                        'score': score,
                        'full_text': ''  # We don't have the full text here
                    })
                    # Remove from missing set to avoid duplicates
                    missing_image_nums.remove(image_id)
            except (ValueError, IndexError):
                continue
    
    # If we still have missing scores, try a simpler approach
    if missing_image_nums:
        # Look for simple patterns like "Image X: 75/100"
        simple_pattern = r'Image\s+(\d+)[^0-9]*(\d{1,3})/?100?'
        simple_matches = re.finditer(simple_pattern, ranking_text, re.IGNORECASE)
        for match in simple_matches:
            try:
                image_id = int(match.group(1))
                score = int(match.group(2))
                if image_id in missing_image_nums and 1 <= score <= 100:
                    results.append({
                        'image_num': image_id,
                        'score': score,
                        'full_text': ''
                    })
                    missing_image_nums.remove(image_id)
            except (ValueError, IndexError):
                continue
    
    return results

def extract_ranking(text):
    """Extract the ranking explanation from text and clean up any redundancies"""
    ranking_patterns = [
        (r'###\s+Ranking(?:\s+Explanation)?:(.*?)(?=###\s+Overall\s+Ranking|###\s+Summary|$)', re.DOTALL),  
        (r'###\s+Ranking\s+Explanation(.*?)(?=###\s+Overall\s+Ranking|###\s+Summary|$)', re.DOTALL),
        (r'\*\*Ranking\s+Explanation:\*\*(.*?)(?=###|\*\*|\n\n\n|RANKING|\Z)', re.DOTALL),  # Updated to catch standalone ranking headers
        (r'###\s+Ranking\s+Summary(.*?)(?=###\s+Conclusion|$)', re.DOTALL),
        (r'###\s+Overall\s+Ranking:(.*?)(?=###\s+Summary|$)', re.DOTALL),
        (r'###\s+Overall\s+Ranking(.*?)(?=###\s+Summary|$)', re.DOTALL),
        (r'###\s+Ranking(.*?)(?=###\s+Summary|$)', re.DOTALL),
        (r'###\s+Ranking\s+and\s+Persuasion\s+Scores:(.*?)(?=###\s+Summary|$)', re.DOTALL),
        (r'###\s+Summary(.*?)(?=###|$)', re.DOTALL),
        (r'\*\*Ranking Summary:\*\*(.*?)(?=\n\n|$)', re.DOTALL),
        (r'Ranking Summary:(.*?)(?=\n\n|$)', re.DOTALL),
        (r'Explanation of Ranking:(.*?)(?=\n\n|$)', re.DOTALL),
        (r'\*\*Reasoning:\*\*(.*?)$', re.DOTALL),  
        (r'Ranking:(.*?)(?=\*\*Reasoning:|Summary|$)', re.DOTALL),
        (r'RANKING(.*?)(?=Summary|$)', re.DOTALL),
        (r'###\s+Ranking\s+of\s+Images(.*?)(?=###\s+Overall\s+Ranking|###\s+Summary|$)', re.DOTALL),
        (r'Ranking\s+Explanation:(.*?)(?=\n\n|RANKING|$)', re.DOTALL),  # Added for cases with text explanation
    ]
    
    # First try to find any explicit ranking sections
    ranking_text = ""
    matched_pattern = None
    
    for pattern, flags in ranking_patterns:
        match = re.search(pattern, text, flags)
        if match:
            ranking_text = match.group(1).strip()
            matched_pattern = pattern
            break
    
    # Special handling for the case with "**Ranking Explanation:**" followed by a "RANKING" header
    if not ranking_text:
        # First check for "**Ranking Explanation:**" pattern
        expl_match = re.search(r'\*\*Ranking\s+Explanation:\*\*(.*?)(?=\n\nRANKING|\Z)', text, re.DOTALL)
        if expl_match:
            ranking_text = expl_match.group(1).strip()
            
            # Check if there's a "RANKING" header after this
            ranking_header_pos = text.find("RANKING", expl_match.end())
            if ranking_header_pos > -1:
                # Get text after RANKING header too
                rest_of_text = text[ranking_header_pos:].strip()
                # Extract content after the RANKING header, until next section or end
                ranking_section_match = re.search(r'RANKING\s*\n-*\s*(.*?)(?=\n\n\n|\Z)', rest_of_text, re.DOTALL)
                if ranking_section_match:
                    extra_text = ranking_section_match.group(1).strip()
                    if extra_text:  # Only add if there's actual content
                        ranking_text += "\n\nRANKING\n" + extra_text
    
    # If we didn't find a ranking text but have a "Ranking Summary" section, use that
    if not ranking_text:
        ranking_summary = re.search(r'\*\*Ranking\s+Summary:\*\*(.*?)(?=\n\n|\Z)', text, re.DOTALL)
        if ranking_summary:
            ranking_text = ranking_summary.group(1).strip()
            matched_pattern = "ranking_summary"
            
            # Look for additional ranking information following the summary
            addition_match = re.search(r'\n\n(.*?)(?=\n\n|\Z)', text[ranking_summary.end():], re.DOTALL)
            if addition_match:
                ranking_text += "\n\n" + addition_match.group(1).strip()
    
    # Look for paragraphs that talk about ranking without a numbered list
    if not ranking_text:
        # Look for paragraphs that mention image rankings
        ranking_paragraph_patterns = [
            r'(Image\s+\d+\s+ranks?\s+higher.*?)(?=\n\n|\Z)',
            r'(The\s+images?\s+are\s+ranked.*?)(?=\n\n|\Z)',
            r'(In\s+terms\s+of\s+ranking.*?)(?=\n\n|\Z)',
            r'(Based\s+on\s+the\s+evaluation.*?rank.*?)(?=\n\n|\Z)',
            r'(Ranking\s+the\s+images.*?)(?=\n\n|\Z)'
        ]
        
        for pattern in ranking_paragraph_patterns:
            para_match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if para_match:
                ranking_text = para_match.group(1).strip()
                matched_pattern = "ranking_paragraph"
                break
    
    # Look for Overall Ranking section separately
    if (not ranking_text or "Overall Ranking" not in ranking_text) and matched_pattern != "ranking_summary":
        overall_ranking = re.search(r'###\s+Overall\s+Ranking:(.*?)(?=###|\Z)', text, re.DOTALL)
        if overall_ranking:
            overall_text = overall_ranking.group(1).strip()
            if ranking_text:
                ranking_text += "\n\n" + overall_text
            else:
                ranking_text = overall_text
    
    # Look for Summary section and add it if found
    if matched_pattern != "ranking_summary" and "Summary" not in ranking_text.lower():
        summary_match = re.search(r'###\s+Summary(.*?)(?=###|\Z)', text, re.DOTALL)
        if summary_match:
            summary_text = summary_match.group(1).strip()
            if ranking_text:
                # Only add summary if it's not already part of the ranking text
                if summary_text.lower() not in ranking_text.lower():
                    ranking_text += "\n\n### Summary" + summary_text
            else:
                ranking_text = "### Summary" + summary_text
        
        # Also look for standalone summary without header
        elif "summary" not in ranking_text.lower():
            summary_match = re.search(r'Summary(.*?)(?=###|\Z)', text, re.DOTALL)
            if summary_match:
                summary_text = summary_match.group(1).strip()
                if ranking_text:
                    # Only add summary if it's not already part of the ranking text
                    if summary_text.lower() not in ranking_text.lower():
                        ranking_text += "\n\nSummary" + summary_text
                else:
                    ranking_text = "Summary" + summary_text
    
    # Look for conclusion section and add it if found
    conclusion_match = re.search(r'###\s+Conclusion(.*?)$', text, re.DOTALL)
    if conclusion_match:
        conclusion_text = conclusion_match.group(1).strip()
        if ranking_text:
            # Only add conclusion if it's not already part of the ranking text
            if conclusion_text.lower() not in ranking_text.lower():
                ranking_text += "\n\n" + conclusion_text
        else:
            ranking_text = conclusion_text
    
    # If no ranking text found using patterns, try to extract it from lists
    if not ranking_text:
        # Look for numbered list with different formats
        list_patterns = [
            r'(\d+\.\s+Image\s+\d+\s+\(Score:\s+\d+\).*?)(?:\n\n|\Z)',
            r'(\d+\.\s+\*\*Image\s+\d+:\*\*\s+\d+/100.*?)(?:\n\n|\Z)'
        ]
        
        for pattern in list_patterns:
            list_match = re.search(pattern, text, re.DOTALL)
            if list_match:
                ranking_text = list_match.group(1).strip()
                break
        
        # If still no ranking text, look for a different format
        if not ranking_text:
            list_match = re.search(r'(\d+\.\s+Image\s+\d+.*?)(?:\n\n|\Z)', text, re.DOTALL)
            if list_match:
                ranking_text = list_match.group(1).strip()

    # If we still don't have ranking text, check for standalone ranking section
    standalone_ranking_pattern = r'###\s+Ranking([\s\S]*?)(?=###\s+Conclusion|\Z)'
    standalone_match = re.search(standalone_ranking_pattern, text)
    if standalone_match:
        ranking_text = standalone_match.group(1).strip()
        
        # Also get the conclusion if it exists
        conclusion_pattern = r'###\s+Conclusion([\s\S]*?)$'
        conclusion_match = re.search(conclusion_pattern, text)
        if conclusion_match:
            conclusion_text = conclusion_match.group(1).strip()
            ranking_text += "\n\n### Conclusion" + conclusion_text
        
        return ranking_text
    # Clean up any redundancies in the ranking text
    ranking_text = clean_redundant_text(ranking_text)
    
    return ranking_text


def normalize_text_format(raw_text):
    """
    Normalize text format by converting bold text headers to standard headers
    and standardizing score formats for easier extraction.
    
    IMPORTANT: Only converts header formats, doesn't modify score values or numbers.
    """
    normalized_text = raw_text
    
    # 1. Convert bold image headers to standard headers without using look-behind
    # First find all "### Image X" or "#### Image X" headers
    existing_headers = re.findall(r'(###|####)\s+Image\s+\d+', normalized_text)
    
    # Then find all "**Image X**" patterns
    bold_headers = re.finditer(r'\*\*Image\s+(\d+)\*\*', normalized_text)
    
    # Convert bold to headers if they aren't already in a header
    for match in reversed(list(bold_headers)):  # Process in reverse to avoid messing up positions
        start, end = match.span()
        image_num = match.group(1)
        # Check if this bold pattern is part of an existing header
        header_prefix = normalized_text[:start].strip().endswith("###") or normalized_text[:start].strip().endswith("####")
        if not header_prefix:
            # Replace with standard header format
            normalized_text = normalized_text[:start] + f"### Image {image_num}" + normalized_text[end:]
    
    # 2. Normalize bullet point formatting - just remove bullets and bold markers, DON'T modify score values
    normalized_text = re.sub(r'-\s+\*\*Persuasion Score:\*\*\s+(\d+)', r'Persuasion Score: \1', normalized_text)
    normalized_text = re.sub(r'-\s+\*\*Persuasion Score\*\*:\s+(\d+)', r'Persuasion Score: \1', normalized_text)
    
    # 3. Standardize formats in ranking section - DON'T modify actual numbers
    normalized_text = re.sub(r'Persuasion Score\s+-\s+(\d+)', r'Persuasion Score: \1', normalized_text)
    
    return normalized_text

def clean_redundant_text(text):
    """Remove redundant sections from the text"""
    if not text:
        return text
        
    # Special handling for duplicate ranking lists
    # Find all occurrences of numbered image lists with scores
    ranking_lists = re.findall(r'(?:\d+\.\s+(?:\*\*)?Image\s+\d+(?:\*\*)?(?::)?\s+(?:\(Score:)?\s*\d+(?:/100)?(?:\))?[^\n]*\n?)+', text)
    
    # If we found multiple ranking lists, keep only the first one
    if len(ranking_lists) > 1:
        for duplicate_list in ranking_lists[1:]:
            text = text.replace(duplicate_list, '', 1)  # Replace only the first occurrence
    
    # Split the text into paragraphs
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    
    # Check for repeated paragraphs
    cleaned_paragraphs = []
    seen_content = set()
    
    for paragraph in paragraphs:
        # Skip short paragraphs that might be headers
        if len(paragraph) < 10:
            cleaned_paragraphs.append(paragraph)
            continue
            
        # Create a normalized version for comparison (strip punctuation, lowercase)
        normalized = re.sub(r'[^\w\s]', '', paragraph.lower())
        normalized = ' '.join(normalized.split())  # Normalize whitespace
        
        # Check if this paragraph contains a ranking list (lines starting with numbers and "Image")
        has_ranking_list = bool(re.search(r'^\d+\.\s+(?:\*\*)?Image\s+\d+', paragraph, re.MULTILINE))
        
        # Special handling for explanatory text prefixed with "**Ranking Explanation:**"
        if paragraph.startswith('**Ranking Explanation:**') or paragraph.startswith('Ranking Explanation:'):
            # Always keep these paragraphs regardless of duplication
            cleaned_paragraphs.append(paragraph)
            continue
        
        # Skip if we've seen this paragraph before (except for headers)
        if normalized in seen_content and not (paragraph.startswith('#') or paragraph.startswith('**')):
            continue
            
        cleaned_paragraphs.append(paragraph)
        
        # Only add to seen content if it's not a heading
        if not (paragraph.startswith('#') or paragraph.startswith('**')):
            seen_content.add(normalized)
    
    # Rebuild the text
    cleaned_text = '\n\n'.join(cleaned_paragraphs)
    
    # One more check for adjacent duplicate ranking lists
    cleaned_text = re.sub(r'((?:\d+\.\s+(?:\*\*)?Image\s+\d+(?:\*\*)?(?::)?\s+(?:\(Score:)?\s*\d+(?:/100)?(?:\))?[^\n]*\n?)+)\s*\n\s*(?:Summary:?\*\*:?)?\s*\n\s*((?:\d+\.\s+(?:\*\*)?Image\s+\d+(?:\*\*)?(?::)?\s+(?:\(Score:)?\s*\d+(?:/100)?(?:\))?[^\n]*\n?)+)', r'\1\n\nSummary:\n', cleaned_text)
    
    return cleaned_text

def verify_data(image_data):
    """Verify that image numbers match and are complete"""
    if not image_data:
        return False
    image_nums = [item['image_num'] for item in image_data]
    if len(set(image_nums)) != len(image_nums):
        print("[WARNING] Duplicate image numbers found!")
        return False
    expected_nums = set(range(1, max(image_nums) + 1))
    actual_nums = set(image_nums)
    if expected_nums != actual_nums:
        print(f"[WARNING] Missing image numbers: {expected_nums - actual_nums}")
        return False
    return True


def main():
    """
    Main function to process dataset and extract generalized image data.
    """
    output_txt_file = "product_evaluations_generalized.txt"
    output_log_file = "extraction_log_generalized.txt"
    
    data = []
    skipped = []
    
    all_categories = sorted(os.listdir(dataset_image))
    selected_categories = all_categories[:25]
    
    with open(output_txt_file, "w", encoding="utf-8") as out_f, \
         open(output_log_file, "w", encoding="utf-8") as log_f:
        
        header = "=== PRODUCT IMAGE EVALUATIONS (GENERALIZED) ===\nExtracted:\n\n"
        out_f.write(header)
        log_f.write(header)
        
        print("\n[INFO] Processing dataset...\n")
        
        for category in selected_categories:
            category_path = os.path.join(dataset_image, category)
            if not os.path.isdir(category_path):
                reason = "Not a directory"
                log_f.write(f"SKIPPED: Category '{category}' - {reason}\n")
                skipped.append({"category": category, "group": None, "reason": reason})
                continue
            
            groups = os.listdir(category_path)
            for group in groups:
                group_path = os.path.join(category_path, group)
                if not os.path.isdir(group_path):
                    continue
                
                images = sorted([
                    os.path.join(group_path, img)
                    for img in os.listdir(group_path)
                    if img.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
                ])
                
                if not images:
                    reason = "No image files found"
                    log_f.write(f"SKIPPED: Category '{category}', Group '{group}' - {reason}\n")
                    skipped.append({"category": category, "group": group, "reason": reason})
                    continue
                
                if len(images) > MAX_IMAGES:
                    reason = f"Too many images ({len(images)} > {MAX_IMAGES})"
                    log_f.write(f"SKIPPED: Category '{category}', Group '{group}' - {reason}\n")
                    skipped.append({"category": category, "group": group, "reason": reason})
                    continue
                
                response_path = os.path.join(dataset_response, category, group, "user_output.txt")
                if not os.path.exists(response_path):
                    reason = "Missing evaluation file"
                    log_f.write(f"SKIPPED: Category '{category}', Group '{group}' - {reason}\n")
                    skipped.append({"category": category, "group": group, "reason": reason})
                    continue
                
                try:
                    image_data, ranking_text = extract_text_data(response_path)
                    
                    if not image_data:
                        reason = "No valid image data extracted"
                        log_f.write(f"SKIPPED: Category '{category}', Group '{group}' - {reason}\n")
                        skipped.append({"category": category, "group": group, "reason": reason})
                        continue
                    
                    data.append({
                        "category": category,
                        "group": group,
                        "images": images,
                        "image_data": image_data,
                        "ranking": ranking_text
                    })
                    
                    out_f.write("=" * 80 + "\n")
                    out_f.write(f"CATEGORY: {category}\n")
                    out_f.write(f"GROUP: {group}\n")
                    out_f.write("=" * 80 + "\n\n")
                    
                    for item in image_data:
                        if item['score'] is None:
                            print(f"[WARNING] Missing score for image {item['image_num']}")
                            continue
                        
                        out_f.write(f"IMAGE {item['image_num']}\n")
                        out_f.write("-" * 40 + "\n")
                        out_f.write(f"Persuasion Score: {item['score']}\n\n")
                        out_f.write(f"Full Text:\n{item['full_text']}\n\n")
                    
                    out_f.write("RANKING\n")
                    out_f.write("-" * 40 + "\n")
                    out_f.write(ranking_text + "\n\n\n")
                    
                    log_f.write(f"SUCCESS: Category '{category}', Group '{group}' - {len(image_data)} images extracted\n")
                    print(f"Processed: {category}/{group} - {len(image_data)} images")
                    
                except Exception as e:
                    reason = f"Error during extraction: {str(e)}"
                    log_f.write(f"ERROR: Category '{category}', Group '{group}' - {reason}\n")
                    skipped.append({"category": category, "group": group, "reason": reason})
                    print(f"Error: {category}/{group} - {str(e)}")
        
        summary = f"\n=== SUMMARY ===\nProcessed: {len(data)} valid entries\nSkipped: {len(skipped)} entries\n"
        out_f.write(summary)
        log_f.write(summary)
    
    print("\nExtraction complete!")
    print(f"Valid entries: {len(data)}")
    print(f"Skipped entries: {len(skipped)}")
    print(f"Output written to {output_txt_file}")
    print(f"Log written to {output_log_file}")

if __name__ == "__main__":
    main()