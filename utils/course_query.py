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

def fuzzy_match_department(dept, course_data, threshold=80):
    all_depts = list(course_data.keys())
    match, score = process.extractOne(dept, all_depts, scorer=fuzz.token_sort_ratio)
    return match if score >= threshold else None

def extract_course_info(user_input, course_data, memory):
    dept_match = re.search(r"(?:in|for|of)?\s*(\b[a-zA-Z ]{3,}\b department)?\s*(computer science|mass communication|nursing|law|biochemistry|architecture)", user_input, re.IGNORECASE)
    level_match = re.search(r"(\d{3}|\d{2,3}-level|\d{3} level|\d{1,3}00|\d{1}-level)", user_input)
    sem_match = re.search(r"(first|1st|second|2nd)\s+semester", user_input.lower())

    department = dept_match.group(2).strip().lower() if dept_match else memory.get("department")
    level = level_match.group(1) if level_match else memory.get("level")
    semester = sem_match.group(1).lower() if sem_match else memory.get("semester")

    # Normalize
    if level:
        level = re.findall(r"\d+", level)[0] + "00"
    if semester:
        semester = "first" if "1" in semester or "first" in semester else "second"

    if department:
        department = fuzzy_match_department(department, course_data)
    else:
        return None

    if not department or department not in course_data:
        return "I couldn't find that department."

    memory["department"] = department
    if level: memory["level"] = level
    if semester: memory["semester"] = semester

    dept_data = course_data[department]
    if level not in dept_data:
        return f"No course data found for {level}-level {department.title()}."

    level_data = dept_data[level]

    if semester:
        sem_key = "1st" if "first" in semester else "2nd"
        courses = level_data.get(sem_key, [])
        if not courses:
            return f"No courses found for {semester} semester."
        return f"Courses for {department.title()} {level}-level ({semester} semester):\n" + "\n".join(
            [f"- {c['code']}: {c['title']} ({c['unit']} units)" for c in courses]
        )
    else:
        # Return both semesters
        courses_1 = level_data.get("1st", [])
        courses_2 = level_data.get("2nd", [])
        response = f"{department.title()} {level}-level courses:\n"
        if courses_1:
            response += "\n📘 First Semester:\n" + "\n".join([f"- {c['code']}: {c['title']} ({c['unit']} units)" for c in courses_1])
        if courses_2:
            response += "\n\n📗 Second Semester:\n" + "\n".join([f"- {c['code']}: {c['title']} ({c['unit']} units)" for c in courses_2])
        return response

def get_course_by_code(user_input, course_data):
    code_match = re.search(r"\b([A-Z]{2,4}\s?\d{3})\b", user_input.upper())
    if not code_match:
        return None

    course_code = code_match.group(1).replace(" ", "").upper()
    for dept, levels in course_data.items():
        for level, semesters in levels.items():
            for sem, courses in semesters.items():
                for course in courses:
                    if course["code"].replace(" ", "").upper() == course_code:
                        return f"{course_code} is **{course['title']}** offered in {dept.title()}, {level}-level ({sem} semester), worth {course['unit']} unit(s)."
    return "I couldn't find any course matching that code."
