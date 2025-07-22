import json
import re
from rapidfuzz import process
from utils.preprocess import normalize_input

# Department to faculty mapping
DEPARTMENT_TO_FACULTY = {
    "Law": "BACOLAW",
    "Mass Communication": "CASMAS",
    "Business Administration": "CASMAS",
    "Economics": "CASMAS",
    "Accounting": "CASMAS",
    "Banking and Finance": "CASMAS",
    "Political Science": "CASMAS",
    "International Relations": "CASMAS",
    "Architecture": "COES",
    "Estate Management": "COES",
    "Urban and Regional Planning": "COES",
    "Computer Science": "CICOT",
    "Information and Communication Technology": "CICOT",
    "Physics": "CONAS",
    "Chemistry": "CONAS",
    "Biochemistry": "CONAS",
    "Microbiology": "CONAS",
    "Mathematics": "CONAS",
    "Anatomy": "COHES",
    "Physiology": "COHES",
    "Nursing": "COHES",
    "Medical Laboratory Science": "COHES",
    "Public Health": "COHES"
}

def fuzzy_match_department(user_input):
    choices = list(DEPARTMENT_TO_FACULTY.keys())
    match, score, _ = process.extractOne(user_input, choices)
    return match if score > 70 else None

def extract_level(user_input):
    match = re.search(r"(100|200|300|400|500)[\s-]*level", user_input)
    return match.group(1) if match else None

def extract_semester(user_input):
    if "first semester" in user_input.lower():
        return "first"
    elif "second semester" in user_input.lower():
        return "second"
    return None  # Both semesters if not specified

def get_courses_by_department_and_level(department, level, semester, course_data):
    dept = department.strip()
    level = level.strip()

    if dept not in course_data:
        return f"❌ No course data found for department: {dept}"

    if level not in course_data[dept]:
        return f"❌ No course data found for {level}-level in {dept}"

    level_data = course_data[dept][level]

    if semester:
        semester = semester.lower()
        if semester not in level_data:
            return f"❌ No {semester} semester data for {level}-level in {dept}"
        courses = level_data[semester]
    else:
        # Both semesters
        courses = []
        for sem in ["first", "second"]:
            courses += level_data.get(sem, [])

    if not courses:
        return f"ℹ️ No courses found for {dept} - {level}-level."

    result = f"📚 Courses for **{dept} - {level}-level**"
    if semester:
        result += f" ({semester.title()} Semester)"
    result += ":\n\n"

    for course in courses:
        code = course.get("code", "")
        title = course.get("title", "")
        unit = course.get("unit", "")
        result += f"- `{code}`: {title} ({unit} unit{'s' if unit != '1' else ''})\n"

    return result

def get_course_info(user_input, course_data):
    normalized = normalize_input(user_input)
    level = extract_level(normalized)
    semester = extract_semester(normalized)

    # Try to find department name
    for dept in DEPARTMENT_TO_FACULTY:
        if dept.lower() in normalized:
            return get_courses_by_department_and_level(dept, level or "100", semester, course_data)

    # Fuzzy match department
    fuzzy_dept = fuzzy_match_department(normalized)
    if fuzzy_dept:
        return get_courses_by_department_and_level(fuzzy_dept, level or "100", semester, course_data)

    # No department, just level mentioned
    if level:
        # Try memory handler fallback in app.py
        return None

    return None  # No useful info extracted
