# utils/course_query.py

import json
import os
from rapidfuzz import fuzz, process

# Load course data
with open("data/course_data.json", "r", encoding="utf-8") as f:
    course_data = json.load(f)

# Department to faculty mapping
department_faculty_map = {
    "computer science": "CICOT",
    "information technology": "CICOT",
    "mass communication": "CASMAS",
    "accounting": "CASMAS",
    "business administration": "CASMAS",
    "architecture": "COES",
    "nursing": "COHES",
    "physiology": "COHES",
    "anatomy": "COHES",
    "physics": "CONAS",
    "chemistry": "CONAS",
    "microbiology": "CONAS",
    "biochemistry": "CONAS",
    "law": "BACOLAW"
}

def fuzzy_match_department(dept_input):
    departments = list(department_faculty_map.keys())
    match, score = process.extractOne(dept_input.lower(), departments, scorer=fuzz.partial_ratio)
    if score > 70:
        return match
    return None

def get_course_info(user_input, memory):
    user_input = user_input.lower()

    # Detect department
    dept = fuzzy_match_department(user_input)
    if dept:
        memory["last_dept"] = dept
    else:
        dept = memory.get("last_dept")

    # Detect level and semester
    level = None
    semester = None

    if "100" in user_input: level = "100"
    elif "200" in user_input: level = "200"
    elif "300" in user_input: level = "300"
    elif "400" in user_input: level = "400"
    elif "500" in user_input: level = "500"

    if "first" in user_input or "1st" in user_input: semester = "first"
    elif "second" in user_input or "2nd" in user_input: semester = "second"

    if not dept or dept not in course_data:
        return None

    result = []
    dept_courses = course_data[dept]

    for course in dept_courses:
        if level and course["level"] != level:
            continue
        if semester and course["semester"].lower() != semester:
            continue
        result.append(f"{course['code']} - {course['title']} ({course['unit']} units)")

    if not result:
        return f"No matching courses found for {dept.title()} with the criteria provided."

    heading = f"Courses for {dept.title()}"
    if level: heading += f", {level}-level"
    if semester: heading += f", {semester} semester"

    return f"📘 {heading}:\n\n" + "\n".join(result)
