import json
import re
from rapidfuzz import fuzz
from utils.preprocess import normalize_input

DEPARTMENTS = {
    "computer science": "College of Information and Communication Technology (CICOT)",
    "mass communication": "College of Information and Communication Technology (CICOT)",
    "political science": "College of Social and Management Sciences (COSMAS)",
    "nursing": "College of Health Sciences (COHES)",
    "law": "College of Law",
    "architecture": "College of Environmental Sciences (COES)",
    "anatomy": "College of Health Sciences (COHES)",
    "physiology": "College of Health Sciences (COHES)",
    "economics with operations research": "College of Social and Management Sciences (COSMAS)",
    "international relations": "College of Social and Management Sciences (COSMAS)",
    "business administration": "College of Social and Management Sciences (COSMAS)",
    "biochemistry": "College of Natural and Applied Sciences (CONAS)",
    "microbiology": "College of Natural and Applied Sciences (CONAS)",
    "chemistry": "College of Natural and Applied Sciences (CONAS)",
    "physics": "College of Natural and Applied Sciences (CONAS)",
    "mathematics": "College of Natural and Applied Sciences (CONAS)",
    "medical laboratory science": "College of Natural and Applied Sciences (CONAS)",
}

LEVEL_KEYWORDS = ["100", "200", "300", "400", "500"]
SEMESTER_KEYWORDS = ["first", "second", "1st", "2nd"]

def fuzzy_match_department(input_text):
    best_match = None
    highest_score = 0
    for dept in DEPARTMENTS:
        score = fuzz.partial_ratio(input_text, dept)
        if score > highest_score and score > 80:
            best_match = dept
            highest_score = score
    return best_match

def parse_query(text):
    text = normalize_input(text)
    level = next((lvl for lvl in LEVEL_KEYWORDS if lvl in text), None)
    semester = next((s for s in SEMESTER_KEYWORDS if s in text), None)
    dept = fuzzy_match_department(text)
    faculty = DEPARTMENTS.get(dept) if dept else None
    return {"department": dept, "faculty": faculty, "level": level, "semester": semester}

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
