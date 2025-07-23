import json
import re
from rapidfuzz import process

# 🔁 Informal input normalization map
NORMALIZATION_MAP = {
    "comp sci": "computer science",
    "mass comm": "mass communication",
    "masscom": "mass communication",
    "nursin": "nursing",
    "nursing science": "nursing",
    "physio": "physiology",
    "microbio": "microbiology",
    "biochem": "biochemistry",
    "biz admin": "business administration",
    "bus admin": "business administration",
    "account": "accounting",
    "law school": "law",
    "pol sci": "political science and international studies",
    "econs": "economics with operations research",
    "arch": "architecture",
    "first sem": "first semester",
    "second sem": "second semester",
    "100lvl": "100 level",
    "200lvl": "200 level",
    "300lvl": "300 level",
    "400lvl": "400 level",
    "wetin": "what",
    "dey": "is",
    "wan": "want",
    "courses dem": "courses",
    "which courses dem dey do": "what are the courses",
    "we dey": "that are",
    "course wey dem dey do": "courses"
}

# ✅ Known departments
DEPARTMENTS = [
    "computer science", "anatomy", "biochemistry", "accounting",
    "business administration", "political science and international studies",
    "microbiology", "economics with operations research", "mass communication",
    "law", "nursing", "physiology", "architecture"
]

# 🏛️ Department to Faculty mapping
DEPARTMENT_TO_FACULTY_MAP = {
    "computer science": "CONAS",
    "anatomy": "COHES",
    "biochemistry": "CONAS",
    "accounting": "CASMAS",
    "business administration": "CASMAS",
    "political science and international studies": "CASMAS",
    "microbiology": "CONAS",
    "economics with operations research": "CASMAS",
    "mass communication": "CASMAS",
    "law": "BACOLAW",
    "nursing": "COHES",
    "physiology": "COHES",
    "architecture": "COES"
}

# 🔤 Normalize text using slang map
def normalize_text(text):
    text = text.lower()
    for slang, std in NORMALIZATION_MAP.items():
        text = text.replace(slang, std)
    return text

# 🎯 Normalize department from input
def normalize_department(text):
    norm_text = normalize_text(text)
    for dept in DEPARTMENTS:
        if dept in norm_text:
            return dept
    return fuzzy_match_department(text)

# 🧠 Fuzzy fallback if department not matched directly
def fuzzy_match_department(text):
    result, score, _ = process.extractOne(text, DEPARTMENTS)
    return result if score >= 80 else None

# 📤 Extract structured course query info
def extract_course_query(text):
    text = normalize_text(text)
    level_match = re.search(r"\b(100|200|300|400)\s*level\b", text)
    semester_match = re.search(r"\b(first|second)\s*semester\b", text)
    department = normalize_department(text)

    return {
        "level": level_match.group(1) if level_match else None,
        "semester": semester_match.group(1).capitalize() if semester_match else None,
        "department": department.title() if department else None,
        "faculty": DEPARTMENT_TO_FACULTY_MAP.get(department) if department else None
    }

# 📂 Load course data from JSON
def load_course_data(path="data/course_data.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# 🧾 Filter and return course(s) by query
def get_courses_for_query(query_info, course_data):
    if not query_info:
        return None

    dept = query_info.get("department", "").lower()
    level = query_info.get("level", "").lower() if query_info.get("level") else None
    semester = query_info.get("semester", "").lower() if query_info.get("semester") else None

    matches = []

    for entry in course_data:
        try:
            if entry["department"].lower() != dept:
                continue
            if level and entry.get("level", "").lower() != level:
                continue
            if semester and semester not in entry.get("question", "").lower():
                continue
            matches.append(entry)
        except KeyError:
            continue

    if not matches:
        return None

    # Return only exact match if possible
    if len(matches) == 1:
        return matches[0]["answer"]

    # Otherwise return cleaned, unique questions
    questions = [f"**{m['question']}**\n{m['answer']}" for m in matches]
    return "\n\n".join(questions)
