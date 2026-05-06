import os
import re
from PIL import Image

# ---------------------------------------------------------------------------------
# Update these paths as needed:
# ---------------------------------------------------------------------------------

## SET PATHS
data_root = "/home/debajyoti/home/debajyoti/debajyoti/product_images_real"
dataset_image = os.path.join(data_root, "dataset_image")
dataset_response = os.path.join(data_root, "dataset_response")
MAX_IMAGES = 4

def find_ranking_position(text):
    """Find the starting position of the ranking section"""
    # First check for explicit ranking headers
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
        r'^(?:\*\*Ranking\s+Summary:\*\*)',
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
    
    # If no explicit header, look for ranking summary pattern
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

    return len(text)  # Default to end of text if no ranking section found

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
    
    # Clean up any redundancies in the ranking text
    ranking_text = clean_redundant_text(ranking_text)
    
    return ranking_text


def extract_score(text):
    """Extract the persuasion score from text using various patterns"""
    score_patterns = [
        r'\*\*Persuasion Score:\*\*\s*\*\*(\d+)/100\*\*',
        r'\*\*Persuasion Score:\*\*\s*(\d+)/100',
        r'\*\*Persuasion Score\*\*:\s*(\d+)/100',
        r'Persuasion Score:\s*(\d+)/100',
        r'-\s+\*\*Persuasion Score\*\*:\s*(\d+)/100',
        r'-\s+Persuasion Score:\s*(\d+)/100',
        r'\*\*Persuasion Score:\*\*\s*(\d+)',
        r'\*\*Persuasion Score\*\*:\s*(\d+)',
        r'Persuasion Score:\s*(\d+)',
        r'-\s+\*\*Persuasion Score\*\*:\s*(\d+)',
        r'-\s+Persuasion Score:\s*(\d+)',
        r'-\s+\*\*Persuasion Score:\*\*\s*(\d+)',
        r'\*\*Persuasion Score:\*\*\s*\*\*(\d+)/100\*\*',
        r'-\s+\*\*Persuasion Score:\*\*\s+(\d+)(?!\s*/)',
        r'Persuasion Score:\*\*\s+(\d+)(?!\s*/)',
        r'\*\*Persuasion Score:\*\*\s+(\d+)(?!\s*/)',
        r'\(Persuasion Score: (\d+)\)',
        r'\*\*Persuasion Score\*\*:\s+(\d+)/100',
        r'\*\*Persuasion Score:\*\*\s+\*\*(\d+)',
        r'\*\*Persuasion Score:\*\*\s+\*\*(\d+)/100',
        r'\*\*Persuasion Score:\*\*\s*(\d+)',
        r'Persuasion Score:\*\*\s*\*\*(\d+)/100',
        r'Persuasion Score:\*\*\s*\*\*(\d+)',
        r'Persuasion Score:\s*\*\*(\d+)/100\*\*',
        r'Persuasion Score:\s*\*\*(\d+)\*\*',
        r'Persuasion Score:\s*(\d+)/100',
        r'Persuasion Score\*\*:\s*(\d+)/100',
        r'Persuasion Score: (\d+)/100',  # Added for the test case format
        r'Persuasion Score.*?(\d+)',  # Fallback pattern
    ]
    
    for pattern in score_patterns:
        score_match = re.search(pattern, text)
        if score_match:
            return int(score_match.group(1))
    
    return None

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


def extract_text_data(file_path):
    """
    Extracts image data and rankings from text files.
    Falls back to extracting scores directly from ranking when no separate image sections exist.
    
    Returns:
        tuple: (image_data, ranking_text)
            - image_data: list of dicts with {'image_num', 'full_text', 'score'}
            - ranking_text: string with the entire ranking explanation
    """
    with open(file_path, "r", encoding="utf-8") as f:
        raw_text = f.read().strip()
    
    print(f"\n[INFO] Processing file: {file_path}")
    
    # First check for "**Ranking Explanation:**" followed by a "RANKING" section
    ranking_expl_match = re.search(r'\*\*Ranking\s+Explanation:\*\*(.*?)(?=\n\nRANKING|\Z)', raw_text, re.DOTALL)
    if ranking_expl_match:
        print("[INFO] Found **Ranking Explanation:** section")
        ranking_position = ranking_expl_match.start()
    else:
        # Check for ranking summary pattern before finding the general ranking position
        ranking_summary_match = re.search(r'(\*\*Ranking\s+Summary:\*\*.*?)(?=\n\n\d+\.\s+\*\*Image|\Z)', raw_text, re.DOTALL)
        
        # Find the position of the ranking section using the find_ranking_position function
        ranking_position = find_ranking_position(raw_text)
        
        # If we found a ranking summary pattern earlier, adjust the ranking position
        if ranking_summary_match and ranking_summary_match.start() < ranking_position:
            print("[INFO] Found ranking summary earlier in text, adjusting extraction")
            ranking_position = ranking_summary_match.start()
    
    # Extract the complete ranking text
    ranking_text = extract_ranking(raw_text[ranking_position:])
    
    # Check for a specific Overall Ranking section that might be after image sections
    overall_ranking_match = re.search(r'###\s+Overall\s+Ranking', raw_text)
    if overall_ranking_match and overall_ranking_match.start() > ranking_position:
        # Only search for image headers in the text before the overall ranking section
        image_text = raw_text[:overall_ranking_match.start()]
    else:
        # Only search for image headers in the text before the ranking section
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
        
        # Handle textual numbers like "First", "Second", etc.
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
            # Check for ranking summary to make sure we don't include it in the image text
            if i+1 < len(all_headers):
                section_end = all_headers[i+1].start()
            else:
                # For the last section, check for specific section delimiters
                overall_ranking_pos = raw_text.find("### Overall Ranking", start_pos)
                ranking_of_images_pos = raw_text.find("### Ranking of Images", start_pos)
                
                # Set the end position to whichever delimiter comes first
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
    processed_image_nums = set()  # Keep track of processed image numbers to avoid duplicates
    
    for section in sections:
        # Skip if we've already processed this image number
        if section['image_num'] in processed_image_nums:
            continue
            
        section_text = image_text[section['start']:section['end']].strip()
        
        # Try to extract score from the section text
        score = extract_score(section_text)
        
        if score is not None:
            image_data.append({
                'image_num': section['image_num'],
                'full_text': section_text,
                'score': score
            })
            processed_image_nums.add(section['image_num'])
    
    # If no image data found or missing scores, try to extract it directly from the ranking text
    # (rest of the function remains unchanged)
    if not image_data or len(image_data) < len(sections):
        missing_image_nums = set(section['image_num'] for section in sections) - processed_image_nums
        print(f"[INFO] Missing scores for images: {missing_image_nums}. Attempting to extract from ranking text.")
        
        # Check for scores in the overall ranking section with format "1. **Image 1:** 85/100"
        ranking_entry_pattern = r'(\d+)\.\s+\*\*Image\s+(\d+):\*\*\s+(\d+)/100'
        ranking_entries = list(re.finditer(ranking_entry_pattern, raw_text, re.MULTILINE))
        
        if ranking_entries:
            for match in ranking_entries:
                try:
                    image_id = int(match.group(2))
                    score = int(match.group(3))
                    
                    if image_id in missing_image_nums:
                        # Get the full text for this image section, if available
                        full_text = ''
                        matching_section = next((s for s in sections if s['image_num'] == image_id), None)
                        if matching_section:
                            full_text = image_text[matching_section['start']:matching_section['end']].strip()
                        
                        image_data.append({
                            'image_num': image_id,
                            'full_text': full_text,
                            'score': score
                        })
                        processed_image_nums.add(image_id)
                        print(f"[INFO] Extracted score {score} for image {image_id} from overall ranking section.")
                except (IndexError, ValueError) as e:
                    print(f"[WARNING] Error extracting from ranking entry match: {e}")
                    continue
        
        # If still missing scores, try the other patterns from the original code
        if missing_image_nums:
            # Look for structured ranking information in the ranking text
            # Try to extract scores from the ranking summary section
            ranking_summary_pattern = r'\d+\.\s+Image\s+(\d+)\s+\(Score:\s+(\d+)\)'
            ranking_entries = list(re.finditer(ranking_summary_pattern, ranking_text, re.MULTILINE))
            
            if not ranking_entries:
                # Try alternative pattern often seen in ranking summaries
                ranking_summary_pattern = r'\d+\.\s+Image\s+(\d+)\s+\((?:Score:)?\s*(\d+)(?:/\d+)?\)'
                ranking_entries = list(re.finditer(ranking_summary_pattern, ranking_text, re.MULTILINE))
                
            if not ranking_entries:
                # Another common pattern
                ranking_summary_pattern = r'\d+\.\s+Image\s+(\d+)(?:\s+\(.*?\))?\s*(?:\(Score:)?\s*(\d+)'
                ranking_entries = list(re.finditer(ranking_summary_pattern, ranking_text, re.MULTILINE))
                
            if not ranking_entries:
                # Look for patterns inside the ranking section like "#### Image X:" or "Image X:" followed by persuasion score
                ranking_image_pattern = r'####\s+Image\s+(\d+):(?:.*?)-\s+\*\*Persuasion\s+Score\*\*:\s*(\d+)/100'
                ranking_entries = list(re.finditer(ranking_image_pattern, raw_text, re.MULTILINE | re.DOTALL))
                
            if not ranking_entries:
                # Try a more specific pattern for the problematic test case
                ranking_image_pattern = r'####\s+Image\s+(\d+):[^#]*?Persuasion\s+Score[^#]*?(\d+)/100'
                ranking_entries = list(re.finditer(ranking_image_pattern, raw_text, re.MULTILINE | re.DOTALL))
            
            # Process the ranking entries to extract scores
            for match in ranking_entries:
                try:
                    image_id = int(match.group(1))
                    score = int(match.group(2))
                    
                    if image_id in missing_image_nums:
                        # Get the full text for this image section, if available
                        full_text = ''
                        matching_section = next((s for s in sections if s['image_num'] == image_id), None)
                        if matching_section:
                            full_text = image_text[matching_section['start']:matching_section['end']].strip()
                        
                        image_data.append({
                            'image_num': image_id,
                            'full_text': full_text,
                            'score': score
                        })
                        processed_image_nums.add(image_id)
                        print(f"[INFO] Extracted score {score} for image {image_id} from ranking summary.")
                except (IndexError, ValueError) as e:
                    print(f"[WARNING] Error extracting from ranking summary match: {e}")
                    continue
                    
            # If we still haven't found all scores, try more patterns in the full text
            still_missing = set(section['image_num'] for section in sections) - processed_image_nums
            if still_missing:
                # Try direct patterns for the specific format in the test case
                for image_id in still_missing:
                    score_pattern = fr'Persuasion Score: (\d+)/100'
                    
                    # Find the image section text for this image
                    matching_section = next((s for s in sections if s['image_num'] == image_id), None)
                    if matching_section:
                        section_text = image_text[matching_section['start']:matching_section['end']].strip()
                        score_match = re.search(score_pattern, section_text)
                        
                        if score_match:
                            try:
                                score = int(score_match.group(1))
                                # Get the full text for this image section, if available
                                full_text = ''
                                matching_section = next((s for s in sections if s['image_num'] == image_id), None)
                                if matching_section:
                                    full_text = image_text[matching_section['start']:matching_section['end']].strip()
                                
                                image_data.append({
                                    'image_num': image_id,
                                    'full_text': full_text,
                                    'score': score
                                })
                                processed_image_nums.add(image_id)
                                print(f"[INFO] Extracted score {score} for image {image_id} from ranking image section.")
                            except (IndexError, ValueError) as e:
                                print(f"[WARNING] Error extracting from image-specific ranking: {e}")
                                continue
    
    # Sort image_data by image_num for consistent output
    image_data.sort(key=lambda x: x['image_num'])
    
    # Verify data integrity
    if not verify_data(image_data) and sections:
        print("[WARNING] Data verification failed, results may be incomplete")
    
    # Print information about extraction
    print(f"[INFO] Extracted {len(image_data)} image entries.")
    print(f"[INFO] Images with valid scores: {sum(1 for item in image_data if item['score'] is not None)}")
    print(f"[INFO] Ranking section length: {len(ranking_text)} characters")
    
    return image_data, ranking_text

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
                
                response_path = os.path.join(dataset_response, category, group, "output_pixtral_zeroshot.txt")
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