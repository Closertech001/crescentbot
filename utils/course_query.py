import json
import re
from rapidfuzz import fuzz
from utils.preprocess import normalize_input

# 🏛️ Mapping from department to faculty/college
DEPARTMENT_TO_FACULTY_MAP = {
    "computer science": "College of Information and Communication Technology (CICOT)",
    "mass communication": "College of Information and Communication Technology (CICOT)",
    "political science and international studies": "College of Arts, Social and Management Sciences (CASMAS)",
    "business administration": "College of Arts, Social and Management Sciences (CASMAS)",
    "economics with operations research": "College of Arts, Social and Management Sciences (CASMAS)",
    "anatomy": "College of Health Sciences (COHES)",
    "physiology": "College of Health Sciences (COHES)",
    "nursing": "College of Health Sciences (COHES)",
    "law": "Bola Ajibola College of Law (BACOLAW)",
    "architecture": "College of Environmental Sciences (COES)",
    "biochemistry": "College of Natural and Applied Sciences (CONAS)",
    "microbiology": "College of Natural and Applied Sciences (CONAS)",
    "chemistry": "College of Natural and Applied Sciences (CONAS)",
    "physics": "College of Natural and Applied Sciences (CONAS)",
    "mathematics": "College of Natural and Applied Sciences (CONAS)",
    "medical laboratory science": "College of Natural and Applied Sciences (CONAS)",
    "accounting": "College of Arts, Social and Management Sciences (CASMAS)"
}

# 📚 Department list
DEPARTMENTS = list(DEPARTMENT_TO_FACULTY_MAP.keys())

# 🎓 Levels and semesters
LEVEL_KEYWORDS = ["100", "200", "300", "400", "500"]
SEMESTER_KEYWORDS = ["first", "second", "1st", "2nd"]

# 🔍 Fuzzy match department
def fuzzy_match_department(input_text):
    best_match = None
    highest_score = 0
    for dept in DEPARTMENTS:
        score = fuzz.partial_ratio(input_text, dept)
        if score > highest_score and score > 80:
            best_match = dept
            highest_score = score
    return best_match

# 🧠 Extract query info
def parse_query(text):
    text = normalize_input(text)
    level = next((lvl for lvl in LEVEL_KEYWORDS if lvl in text), None)
    semester = next((s for s in SEMESTER_KEYWORDS if s in text), None)
    dept = fuzzy_match_department(text)
    faculty = DEPARTMENT_TO_FACULTY_MAP.get(dept) if dept else None
    return {
        "department": dept,
        "faculty": faculty,
        "level": level,
        "semester": "First" if semester in ["first", "1st"] else "Second" if semester in ["second", "2nd"] else None
    }

# 📦 Get courses from structured course_data
def get_courses_for_query(course_data, query_info):
    dept = query_info.get("department")
    level = query_info.get("level")
    semester = query_info.get("semester")

    matches = []
    for entry in course_data:
        if dept and dept.lower() != entry["department"].lower():
            continue
        if level and level != entry.get("level"):
            continue
        if semester and semester.lower() not in entry.get("question", "").lower():
            continue
        matches.append(entry)

    if not matches and dept:
        matches = [entry for entry in course_data if dept.lower() in entry["department"].lower()]

    return matches
