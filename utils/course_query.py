import json
import re
from rapidfuzz import process

# Load course data from the correct path
with open("data/course_data.json", "r", encoding="utf-8") as f:
    COURSE_DATA = json.load(f)

# Mapping department to faculty
DEPARTMENT_TO_FACULTY = {
    "mass communication": "CASMAS",
    "economics": "CASMAS",
    "political science": "CASMAS",
    "accounting": "CASMAS",
    "banking and finance": "CASMAS",
    "business administration": "CASMAS",
    "architecture": "COES",
    "estate management": "COES",
    "computer science": "CICOT",
    "information technology": "CICOT",
    "biology": "CONAS",
    "chemistry": "CONAS",
    "microbiology": "CONAS",
    "mathematics": "CONAS",
    "physics": "CONAS",
    "biochemistry": "COHES",
    "nursing": "COHES",
    "anatomy": "COHES",
    "physiology": "COHES",
    "law": "BACOLAW"
}

def extract_course_query(user_input):
    """Extract department, level, and optional semester from user input."""
    department = None
    level = None
    semester = None

    # Attempt to find level (e.g., 100, 200, etc.)
    level_match = re.search(r"\b(100|200|300|400|500)\s?level\b", user_input)
    if level_match:
        level = level_match.group(1)

    # Attempt to find semester
    semester_match = re.search(r"\b(first|1st|second|2nd)\s+semester\b", user_input)
    if semester_match:
        sem_raw = semester_match.group(1)
        semester = "first" if "1" in sem_raw or "first" in sem_raw else "second"

    # Fuzzy match department
    department_names = list(DEPARTMENT_TO_FACULTY.keys())
    match = process.extractOne(user_input, department_names, score_cutoff=60)
    if match:
        department = match[0]

    return department, level, semester

def get_courses_for_query(department, level=None, semester=None):
    """Return matching courses from structured course data."""
    faculty = DEPARTMENT_TO_FACULTY.get(department.lower())
    if not faculty:
        return f"Sorry, I couldn't identify the faculty for {department}. Please check the department name."

    try:
        courses = COURSE_DATA[faculty][department.lower()]
    except KeyError:
        return f"Sorry, I couldn't find courses for {department}."

    # Filter by level
    if level:
        courses = courses.get(level, {})
        if not courses:
            return f"No courses found for {department} at {level}-level."
    else:
        # If no level specified, return all levels
        all_courses = []
        for lvl, lv_courses in courses.items():
            all_courses.append(f"\n📘 {lvl}-level courses:")
            for sem in lv_courses:
                all_courses.append(f"  ➤ {sem.capitalize()} Semester:")
                for course in lv_courses[sem]:
                    all_courses.append(f"    • {course['code']} - {course['title']} ({course['unit']} units)")
        return "\n".join(all_courses)

    # Filter by semester
    if semester:
        courses = courses.get(semester, [])
        if not courses:
            return f"No {semester} semester courses found for {department} at {level}-level."

        return "\n".join([f"• {c['code']} - {c['title']} ({c['unit']} units)" for c in courses])

    # If only level is provided
    result = []
    for sem, sem_courses in courses.items():
        result.append(f"\n📗 {sem.capitalize()} Semester:")
        for course in sem_courses:
            result.append(f"• {course['code']} - {course['title']} ({course['unit']} units)")
    return "\n".join(result)
