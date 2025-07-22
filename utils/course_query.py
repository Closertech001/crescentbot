import json
import re
from rapidfuzz import fuzz
from utils.preprocess import normalize_input

# 🏛️ Department → Faculty map
DEPARTMENT_TO_FACULTY_MAP = {
    "computer science": "College of Information and Communication Technology (CICOT)",
    "mass communication": "College of Information and Communication Technology (CICOT)",
    "political science and international studies": "College of Arts, Social and Management Sciences (CASMAS)",
    "business administration": "College of Arts, Social and Management Sciences (CASMAS)",
    "economics with operations research": "College of Arts, Social and Management Sciences (CASMAS)",
    "accounting": "College of Arts, Social and Management Sciences (CASMAS)",
    "architecture": "College of Environmental Sciences (COES)",
    "anatomy": "College of Health Sciences (COHES)",
    "physiology": "College of Health Sciences (COHES)",
    "nursing": "College of Health Sciences (COHES)",
    "law": "Bola Ajibola College of Law (BACOLAW)",
    "biochemistry": "College of Natural and Applied Sciences (CONAS)",
    "microbiology": "College of Natural and Applied Sciences (CONAS)",
    "chemistry": "College of Natural and Applied Sciences (CONAS)",
    "physics": "College of Natural and Applied Sciences (CONAS)",
    "mathematics": "College of Natural and Applied Sciences (CONAS)",
    "medical laboratory science": "College of Natural and Applied Sciences (CONAS)",
}

DEPARTMENTS = list(DEPARTMENT_TO_FACULTY_MAP.keys())
LEVEL_KEYWORDS = ["100", "200", "300", "400", "500", "year 1", "year 2", "year 3", "year 4", "year 5", "final year"]
SEMESTER_KEYWORDS = ["first", "second", "1st", "2nd"]

# 🔍 Fuzzy match multiple departments in a query
def fuzzy_match_departments(text):
    matches = []
    for dept in DEPARTMENTS:
        score = fuzz.partial_ratio(text.lower(), dept.lower())
        if score > 80:
            matches.append(dept)
    return list(set(matches))

# 📘 Level normalization
def extract_level(text):
    if "final year" in text:
        return "400"  # or "500" depending on department
    if "year 1" in text:
        return "100"
    elif "year 2" in text:
        return "200"
    elif "year 3" in text:
        return "300"
    elif "year 4" in text:
        return "400"
    elif "year 5" in text:
        return "500"
    for lvl in ["100", "200", "300", "400", "500"]:
        if lvl in text:
            return lvl
    return None

# 📘 Semester normalization
def extract_semester(text):
    if "first" in text or "1st" in text:
        return "First"
    elif "second" in text or "2nd" in text:
        return "Second"
    return None

# 🧠 Parse query into department, level, semester, faculty
def parse_query(text):
    text = normalize_input(text)
    level = extract_level(text)
    semester = extract_semester(text)
    departments = fuzzy_match_departments(text)
    faculties = [DEPARTMENT_TO_FACULTY_MAP.get(d) for d in departments]

    return {
        "departments": departments,
        "faculties": faculties,
        "level": level,
        "semester": semester
    }

# 📦 Match course info based on query
def get_courses_for_query(query_info, course_data):
    departments = query_info.get("departments", [])
    level = query_info.get("level")
    semester = query_info.get("semester")

    matches = []
    for entry in course_data:
        if departments and entry["department"].lower() not in [d.lower() for d in departments]:
            continue
        if level and level != entry.get("level"):
            continue
        if semester and semester.lower() not in entry.get("question", "").lower():
            continue
        matches.append(entry)

    # fallback if nothing matched by level/semester
    if not matches and departments:
        matches = [
            entry for entry in course_data
            if entry["department"].lower() in [d.lower() for d in departments]
        ]

    return matches

# ✅ Final function to use in app.py
def get_course_info(user_query, course_data):
    query_info = parse_query(user_query)
    return get_courses_for_query(query_info, course_data)
