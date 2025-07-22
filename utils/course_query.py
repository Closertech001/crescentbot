import json
import re
from rapidfuzz import fuzz
from utils.preprocess import normalize_input

# 🏛️ Department → Faculty/College map
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

LEVEL_KEYWORDS = ["100", "200", "300", "400", "500"]
SEMESTER_KEYWORDS = ["first", "second", "1st", "2nd"]

# 🔍 Fuzzy matcher for multiple departments
def fuzzy_match_departments(text):
    """
    Match multiple departments from input using fuzzy matching.
    Returns a list of department names.
    """
    matches = []
    for dept in DEPARTMENTS:
        score = fuzz.partial_ratio(text.lower(), dept.lower())
        if score > 80:
            matches.append(dept)
    return list(set(matches))  # remove duplicates

# 🧠 Query parser
def parse_query(text):
    text = normalize_input(text)
    level = next((lvl for lvl in LEVEL_KEYWORDS if lvl in text), None)
    semester = next((s for s in SEMESTER_KEYWORDS if s in text), None)
    departments = fuzzy_match_departments(text)
    faculties = [DEPARTMENT_TO_FACULTY_MAP.get(d) for d in departments]

    return {
        "departments": departments,
        "faculties": faculties,
        "level": level,
        "semester": "First" if semester in ["first", "1st"] else "Second" if semester in ["second", "2nd"] else None
    }

# 📦 Course fetcher
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

    # fallback: if level/semester filtering fails but department matches
    if not matches and departments:
        matches = [
            entry for entry in course_data
            if entry["department"].lower() in [d.lower() for d in departments]
        ]

    return matches
