import json
import re
from rapidfuzz import process, fuzz

# Mapping of departments to faculties
DEPARTMENT_TO_FACULTY = {
    "anatomy": "COHES",
    "nursing": "COHES",
    "physiology": "COHES",
    "architecture": "COES",
    "computer science": "CICOT",
    "information technology": "CICOT",
    "physics": "CONAS",
    "microbiology": "CONAS",
    "mass communication": "CASMAS",
    "political science": "CASMAS",
    "economics": "CASMAS",
    "law": "BACOLAW",
    "accounting": "CASMAS",
    # Add more mappings as needed
}

def fuzzy_match_department(user_input, threshold=70):
    user_input = user_input.lower()
    best_match, score = process.extractOne(user_input, DEPARTMENT_TO_FACULTY.keys(), scorer=fuzz.ratio)
    return best_match if score >= threshold else None

def get_course_info(user_input, course_data_path="data/course_data.json"):
    try:
        with open(course_data_path, "r", encoding="utf-8") as f:
            course_data = json.load(f)
    except Exception:
        return None

    user_input = user_input.lower()
    matched_dept = fuzzy_match_department(user_input)

    level_match = re.search(r"\b(100|200|300|400|500)\b", user_input)
    semester_match = re.search(r"\b(first|second)\s*semester\b", user_input)

    level = f"{level_match.group(1)}" if level_match else None
    semester = semester_match.group(1).lower() if semester_match else None

    # Specific course code
    code_match = re.search(r"\b([A-Z]{2,4}\s?\d{3})\b", user_input.upper())
    if code_match:
        code = code_match.group(1).replace(" ", "")
        for faculty in course_data.values():
            for department, levels in faculty.items():
                for lvl, semesters in levels.items():
                    for sem, courses in semesters.items():
                        for course in courses:
                            if course["code"].replace(" ", "").upper() == code:
                                return f'{course["code"]}: {course["title"]} ({course["unit"]} units)'
        return "Sorry, I couldn’t find that course code."

    # Full course listing
    if matched_dept:
        faculty_key = DEPARTMENT_TO_FACULTY.get(matched_dept)
        department_data = course_data.get(faculty_key, {}).get(matched_dept)

        if not department_data:
            return "I couldn’t find courses for that department."

        if level and semester:
            courses = department_data.get(level, {}).get(semester)
            if not courses:
                return f"No courses found for {matched_dept} {level} {semester} semester."
            return "\n".join(f'{c["code"]}: {c["title"]} ({c["unit"]} units)' for c in courses)

        elif level:
            response = []
            for sem, courses in department_data.get(level, {}).items():
                if courses:
                    course_lines = [f'{c["code"]}: {c["title"]} ({c["unit"]} units)' for c in courses]
                    response.append(f"--- {sem.capitalize()} Semester ---\n" + "\n".join(course_lines))
            return "\n\n".join(response) if response else f"No courses found for {matched_dept} {level} level."

        else:
            response = []
            for lvl, sems in department_data.items():
                for sem, courses in sems.items():
                    if courses:
                        course_lines = [f'{c["code"]}: {c["title"]} ({c["unit"]} units)' for c in courses]
                        response.append(f"{lvl} - {sem.capitalize()} Semester\n" + "\n".join(course_lines))
            return "\n\n".join(response) if response else f"No courses found for {matched_dept}."

    return None
