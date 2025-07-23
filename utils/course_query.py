# utils/course_query.py

import json
import re
from rapidfuzz import process, fuzz

with open("course_data.json", "r", encoding="utf-8") as f:
    course_data = json.load(f)

def extract_level_semester(text):
    level_match = re.search(r"\b(100|200|300|400|500)\s*level\b", text)
    semester_match = re.search(r"\b(first|second)\s*semester\b", text)
    return (
        level_match.group(1) if level_match else None,
        semester_match.group(1) if semester_match else None,
    )

def match_department(user_input):
    all_departments = list(course_data.keys())
    best_match, score = process.extractOne(user_input, all_departments, scorer=fuzz.token_sort_ratio)
    return best_match if score > 70 else None

def get_course_info(user_input):
    level, semester = extract_level_semester(user_input)
    dept = match_department(user_input)

    if not dept:
        return "🤔 I couldn't find the department you're referring to. Please check and try again."

    courses = course_data.get(dept, {}).get("courses", {})
    if level:
        courses = courses.get(level, {})
        if semester:
            courses = courses.get(semester, [])

    if isinstance(courses, dict):  # In case level but no semester
        all_courses = []
        for sem in courses.values():
            all_courses.extend(sem)
        courses = all_courses
    elif isinstance(courses, list):
        pass
    else:
        return "🤷‍♂️ I couldn't find courses matching your request."

    if not courses:
        return "📚 No course data found for your query."

    formatted = "\n".join([f"- {c['code']} — {c['title']} ({c['unit']} unit{'s' if c['unit'] != 1 else ''})" for c in courses])
    return f"Here are the courses for **{dept}**{f', {level} level' if level else ''}{f', {semester} semester' if semester else ''}:\n\n{formatted}"
