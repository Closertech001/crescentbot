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

def normalize_text(text: str) -> str:
    """Normalize slang and informal words in input text."""
    text = text.lower()
    for slang, standard in NORMALIZATION_MAP.items():
        text = text.replace(slang, standard)
    return text

def fuzzy_match_department(text: str) -> str | None:
    """Use fuzzy matching to find closest department if no direct match."""
    result, score, _ = process.extractOne(text, DEPARTMENTS)
    return result if score >= 80 else None

def normalize_department(text: str) -> str | None:
    """Normalize input and detect department."""
    norm_text = normalize_text(text)
    for dept in DEPARTMENTS:
        if dept in norm_text:
            return dept
    return fuzzy_match_department(norm_text)

def extract_course_query(text: str) -> dict:
    """Extract department, level, semester from input text."""
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

def load_course_data(path="data/crescent_qa.json") -> list:
    """Load course data from JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def get_courses_for_query(query_info: dict, course_data: list) -> str | None:
    """Return all relevant course answers matching query."""
    if not query_info:
        return None

    dept = query_info.get("department", "").lower()
    level = query_info.get("level", "").lower() if query_info.get("level") else None
    semester = query_info.get("semester", "").lower() if query_info.get("semester") else None

    results = []

    for entry in course_data:
        try:
            if entry.get("department", "").lower() != dept:
                continue
            if level and entry.get("level", "").lower() != level:
                continue
            # For semester, check if it's in the entry either as a field or in question text
            entry_semester = entry.get("semester", "").lower()
            question_text = entry.get("question", "").lower()
            if semester and semester not in entry_semester and semester not in question_text:
                continue
            results.append(entry.get("answer", "").strip())
        except Exception:
            continue

    if results:
        # Return joined string with numbering for clarity
        return "\n\n".join(f"{idx+1}. {ans}" for idx, ans in enumerate(results))
    else:
        return None
