# utils/course_query.py

import json
from rapidfuzz import process, fuzz

# Load course data
with open("data/course_data.json", "r", encoding="utf-8") as f:
    course_data = json.load(f)

# List of all departments in the data
all_departments = list(course_data.keys())

def match_department(input_dept):
    """Fuzzy match user department to closest real department."""
    if not input_dept:
        return None
    match, score, _ = process.extractOne(input_dept, all_departments, scorer=fuzz.token_sort_ratio)
    return match if score >= 70 else None

def extract_course_info(department=None, level=None, semester=None):
    """
    Extract course info based on fuzzy department, level (100, 200...), and semester (first/second).
    All parameters are optional.
    """
    # Normalize inputs
    department = match_department(department) if department else None
    level = str(level) if level else None
    semester = semester.lower().strip() if semester else None

    results = []

    for dept_name, level_dict in course_data.items():
        if department and dept_name != department:
            continue

        for level_key, semester_dict in level_dict.items():
            if level and level_key != level:
                continue

            for sem_key, courses in semester_dict.items():
                if semester and semester not in sem_key.lower():
                    continue

                for course in courses:
                    results.append({
                        "department": dept_name,
                        "level": level_key,
                        "semester": sem_key,
                        "course_code": course.get("code"),
                        "course_title": course.get("title"),
                        "unit": course.get("unit"),
                    })

    return results
