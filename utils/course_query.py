# utils/course_query.py

import json
import re

# Extract department, level, and semester from query
def extract_course_query(text):
    text = text.lower()
    level_match = re.search(r"\b(100|200|300|400)\s*level\b", text)
    semester_match = re.search(r"\b(first|second)\s*semester\b", text)

    departments = [
        "computer science", "anatomy", "biochemistry", "accounting",
        "business administration", "political science and international studies",
        "microbiology", "economics with operations research", "mass communication",
        "law", "nursing", "physiology", "architecture"
    ]
    department_match = next((d for d in departments if d in text), None)

    return {
        "level": level_match.group(1) if level_match else None,
        "semester": semester_match.group(1).capitalize() if semester_match else None,
        "department": department_match.title() if department_match else None
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
