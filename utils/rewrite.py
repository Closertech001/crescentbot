import re

# Optional: could be extended to support pronoun resolution, etc.
def rewrite_followup(current_input, last_query_info):
    if not last_query_info:
        return current_input

    rewritten = current_input

    if "level" in last_query_info and "level" not in current_input:
        rewritten += f" for {last_query_info['level']} level"
    if "department" in last_query_info and "department" not in current_input:
        rewritten += f" in {last_query_info['department']} department"

    return rewritten
# utils/rewrite.py

import re

def clean_input(text: str) -> str:
    """
    Basic cleanup of the user's input:
    - Removes extra spaces
    - Strips whitespace
    - Removes emojis and special characters except alphanumerics and question marks
    """
    text = re.sub(r'[^\w\s\?]', '', text)  # Keep letters, digits, spaces, and question marks
    text = re.sub(r'\s+', ' ', text)       # Remove multiple spaces
    return text.strip().lower()


def standardize_question(text: str) -> str:
    """
    Standardizes input for improved matching. Examples:
    - 'tell me about CSC101' → 'what is CSC101?'
    - 'info on ARC 205' → 'what is ARC 205?'
    """
    text = clean_input(text)

    # Match course code patterns like CSC101, ARC 205, etc.
    course_match = re.search(r'\b([a-zA-Z]{2,4})\s?(\d{3})\b', text)
    if course_match:
        code = f"{course_match.group(1).upper()} {course_match.group(2)}"
        return f"what is {code}?"

    # Generic fallback
    return text
