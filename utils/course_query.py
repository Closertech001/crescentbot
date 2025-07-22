# utils/course_query.py

import json
import re

def extract_course_query(user_input):
    """
    Extracts level, semester, and department from user input using regex.
    Example: "What are 200 level second semester courses in Law?"
    """
    level_match = re.search(r"\b(100|200|300|400|500)\s*level\b", user_input, re.IGNORECASE)
    semester_match = re.search(r"\b(first|1st|second|2nd)\s+semester\b", user_input, re.IGNORECASE)
    dept_match = re.search(r"in\s+(.*?)(?:\?|$)", user_input, re.IGNORECASE)

    level = f"{level_match.group(1)} level" if level_match else None
    semester = semester_match.group(1).lower() if semester_match else None
    department = dept_match.group(1).strip().title() if dept_match else None

    if semester in ["1st", "first"]:
        semester = "First"
    elif semester in ["2nd", "second"]:
        semester = "Second"

    return {
        "level": level,
        "semester": semester,
        "department": department
    }

def get_course_info(course_data_path, level=None, semester=None, department=None):
    with open(course_data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    filtered = []
    for entry in data:
        if level and entry.get("level", "").lower() != level.lower():
            continue
        if semester and entry.get("semester", "").lower() != semester.lower():
            continue
        if department and entry.get("department", "").lower() != department.lower():
            continue
        filtered.append(entry)

    return filtered
