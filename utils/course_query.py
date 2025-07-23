import json
import re
from rapidfuzz import process

# Department → Faculty map
department_to_faculty = {
    "law": "Bola Ajibola College of Law",
    "mass communication": "College of Arts, Social and Management Sciences",
    "economics": "College of Arts, Social and Management Sciences",
    "accounting": "College of Arts, Social and Management Sciences",
    "architecture": "College of Environmental Sciences",
    "computer science": "College of Information and Communication Technology",
    "nursing": "College of Health Sciences",
    "microbiology": "College of Natural and Applied Sciences",
    "chemistry": "College of Natural and Applied Sciences",
    "physics": "College of Natural and Applied Sciences",
    "anatomy": "College of Health Sciences",
    # Add others as needed
}

def fuzzy_match_department(user_input):
    choices = list(department_to_faculty.keys())
    match, score = process.extractOne(user_input.lower(), choices)
    return match if score >= 70 else None

def extract_course_query(text):
    query = {
        "department": None,
        "level": None,
        "semester": None,
        "course_code": None
    }

    # Course code
    code_match = re.search(r"\b([A-Z]{2,4})\s?(\d{3})\b", text.upper())
    if code_match:
        query["course_code"] = code_match.group(1) + " " + code_match.group(2)

    # Level (e.g. 100, 200)
    level_match = re.search(r"\b(100|200|300|400|500|600)\s?(level)?\b", text)
    if level_match:
        query["level"] = level_match.group(1)

    # Semester
    if "first semester" in text or "1st semester" in text:
        query["semester"] = "first"
    elif "second semester" in text or "2nd semester" in text:
        query["semester"] = "second"

    # Department fuzzy match
    for dept in department_to_faculty:
        if dept in text.lower():
            query["department"] = dept
            break
    if not query["department"]:
        query["department"] = fuzzy_match_department(text)

    return query

def get_courses_for_query(query, course_data):
    dept = query.get("department")
    level = query.get("level")
    semester = query.get("semester")
    code = query.get("course_code")

    if not any([dept, level, semester, code]):
        return None

    result = []

    for course in course_data:
        match = True
        if dept and dept.lower() not in course["department"].lower():
            match = False
        if level and str(course.get("level")) != str(level):
            match = False
        if semester and semester.lower() != course.get("semester", "").lower():
            match = False
        if code and code.upper() != course.get("code", "").upper():
            match = False

        if match:
            result.append(f'{course["code"]} - {course["title"]} ({course["unit"]} unit{"s" if course["unit"] != 1 else ""})')

    if result:
        header = ""
        if code:
            header = f"Here’s the detail for `{code}`:"
        elif dept:
            header = f"Courses for {dept.title()}"
            if level:
                header += f" {level} level"
            if semester:
                header += f" ({semester} semester)"
            header += ":"
        return f"{header}\n\n" + "\n".join(result)

    return None
