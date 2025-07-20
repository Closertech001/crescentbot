# utils/course_query.py

import json
import re

# Maps for common department abbreviations and synonyms
DEPARTMENT_MAP = {
    "comp sci": "computer science",
    "cs": "computer science",
    "mass comm": "mass communication",
    "masscom": "mass communication",
    "nursin": "nursing",
    "nursing science": "nursing",
    "physio": "physiology",
    "microbio": "microbiology",
    "micro biology": "microbiology",
    "biochem": "biochemistry",
    "biz admin": "business administration",
    "bus admin": "business administration",
    "account": "accounting",
    "law school": "law",
    "pol sci": "political science and international studies",
    "political science": "political science and international studies",
    "econs": "economics with operations research",
    "economics": "economics with operations research",
    "arch": "architecture"
}

# Full list of supported departments
DEPARTMENTS = [
    "computer science", "anatomy", "biochemistry", "accounting",
    "business administration", "political science and international studies",
    "microbiology", "economics with operations research", "mass communication",
    "law", "nursing", "physiology", "architecture"
]

# Normalize and map user input to known department name
def normalize_department(text):
    text = text.lower()
    for keyword, standard in DEPARTMENT_MAP.items():
        if keyword in text:
            return standard
    for dept in DEPARTMENTS:
        if dept in text:
            return dept
    return None

# Extract department, level, and semester from query
def extract_course_query(text):
    text = text.lower()
    level_match = re.search(r"\b(100|200|300|400)\s*level\b", text)
    semester_match = re.search(r"\b(first|second)\s*semester\b", text)
    department = normalize_department(text)

    return {
        "level": level_match.group(1) if level_match else None,
        "semester": semester_match.group(1).capitalize() if semester_match else None,
        "department": department.title() if department else None
    }

# Load course data JSON file
def load_course_data(path="data/course_data.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# Find matching courses based on extracted info
def get_courses_for_query(query_info, course_data):
    for entry in course_data:
        if (
            entry["department"].lower() == query_info["department"].lower()
            and entry["level"].lower() == query_info["level"].lower()
            and entry["question"].lower().find(query_info["semester"].lower()) != -1
        ):
            return entry["answer"]
    return None
