# utils/course_query.py

import json
import re

# Map informal phrases/slang/abbreviations to standardized terms
NORMALIZATION_MAP = {
    "comp sci": "computer science",
    "mass comm": "mass communication",
    "masscom": "mass communication",
    "nursin": "nursing",
    "nursing science": "nursing",
    "physio": "physiology",
    "microbio": "microbiology",
    "biochem": "biochemistry",
    "biz admin": "business administration",
    "bus admin": "business administration",
    "account": "accounting",
    "law school": "law",
    "pol sci": "political science and international studies",
    "econs": "economics with operations research",
    "arch": "architecture",
    "first sem": "first semester",
    "second sem": "second semester",
    "100lvl": "100 level",
    "200lvl": "200 level",
    "300lvl": "300 level",
    "400lvl": "400 level",
    "wetin": "what",
    "dey": "is",
    "wan": "want",
    "courses dem": "courses",
    "which courses dem dey do": "what are the courses",
    "we dey": "that are",
    "course wey dem dey do": "courses"
}

DEPARTMENTS = [
    "computer science", "anatomy", "biochemistry", "accounting",
    "business administration", "political science and international studies",
    "microbiology", "economics with operations research", "mass communication",
    "law", "nursing", "physiology", "architecture"
]

# Normalize input text using the maps
def normalize_text(text):
    text = text.lower()
    for key, val in NORMALIZATION_MAP.items():
        text = text.replace(key, val)
    return text

def normalize_department(text):
    for keyword, standard in NORMALIZATION_MAP.items():
        if keyword in text and standard in DEPARTMENTS:
            return standard
    for dept in DEPARTMENTS:
        if dept in text:
            return dept
    return None

# Extract department, level, semester from input
def extract_course_query(text):
    text = normalize_text(text)
    level_match = re.search(r"\b(100|200|300|400)\s*level\b", text)
    semester_match = re.search(r"\b(first|second)\s*semester\b", text)
    department = normalize_department(text)

    return {
        "level": level_match.group(1) if level_match else None,
        "semester": semester_match.group(1).capitalize() if semester_match else None,
        "department": department.title() if department else None
    }

# Load course data
def load_course_data(path="data/course_data.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# Fetch matching course info
def get_courses_for_query(query_info, course_data):
    for entry in course_data:
        if (
            entry["department"].lower() == query_info["department"].lower()
            and entry["level"].lower() == query_info["level"].lower()
            and entry["question"].lower().find(query_info["semester"].lower()) != -1
        ):
            return entry["answer"]
    return None
