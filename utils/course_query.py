import json
import re
from rapidfuzz import fuzz, process

# Load course data
with open("data/course_data.json", "r", encoding="utf-8") as f:
    COURSE_DATA = json.load(f)

# Mapping of departments to their colleges
DEPARTMENT_TO_COLLEGE = {
    "computer science": "CICOT",
    "information technology": "CICOT",
    "mass communication": "CASMAS",
    "political science": "CASMAS",
    "criminology and security studies": "CASMAS",
    "accounting": "CASMAS",
    "banking and finance": "CASMAS",
    "economics": "CASMAS",
    "business administration": "CASMAS",
    "architecture": "COES",
    "estate management": "COES",
    "law": "BACOLAW",
    "microbiology": "CONAS",
    "biochemistry": "CONAS",
    "chemistry": "CONAS",
    "computer science (conas)": "CONAS",
    "mathematics": "CONAS",
    "physics": "CONAS",
    "nursing": "COHES",
    "anatomy": "COHES",
    "physiology": "COHES",
    "medical laboratory science": "COHES",
}

def normalize_department_name(name):
    name = name.lower().strip()
    best_match, score, _ = process.extractOne(
        name, list(DEPARTMENT_TO_COLLEGE.keys()), scorer=fuzz.ratio
    )
    return best_match if score > 70 else None

def extract_course_info(user_input):
    user_input = user_input.lower()

    # Extract level
    level_match = re.search(r"\b(100|200|300|400|500)\s*(?:level)?\b", user_input)
    level = level_match.group(1) if level_match else None

    # Extract semester
    semester_match = re.search(r"\b(first|1st|second|2nd)\s+semester\b", user_input)
    semester = (
        "first"
        if semester_match and "1" in semester_match.group(1)
        else "second"
        if semester_match
        else None
    )

    # Extract department
    dept_match = re.search(
        r"\b(?:course[s]?|subject[s]?)\s+(?:for|in|of)\s+([a-zA-Z &]+)", user_input
    )
    dept_name = dept_match.group(1).strip() if dept_match else None
    if not dept_name:
        # Try to extract department directly
        for dept in DEPARTMENT_TO_COLLEGE:
            if dept in user_input:
                dept_name = dept
                break

    normalized_dept = normalize_department_name(dept_name or "")
    if not normalized_dept:
        return "I couldn't identify the department you're referring to. Please try again."

    college = DEPARTMENT_TO_COLLEGE[normalized_dept]
    courses = COURSE_DATA.get(college, {}).get(normalized_dept, {}).get("courses", {})

    # Apply filters
    filtered = []
    for course in courses:
        if level and str(course.get("level")) != str(level):
            continue
        if semester and course.get("semester", "").lower() != semester:
            continue
        filtered.append(course)

    if not filtered:
        return f"No courses found for {normalized_dept.title()} at {level or 'any'} level {f'in the {semester} semester' if semester else ''}."

    response = f"Courses for **{normalized_dept.title()}**"
    if level:
        response += f" - {level}-level"
    if semester:
        response += f" ({semester.title()} Semester)"
    response += ":\n\n"

    for course in filtered:
        response += f"- `{course['code']}`: {course['title']} ({course['unit']} units)\n"

    return response
