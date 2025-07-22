# utils/course_query.py

import json
import re
from utils.memory import MemoryHandler

# Load structured course data
with open("data/course_data.json", "r", encoding="utf-8") as f:
    course_data = json.load(f)


def extract_course_info(user_input: str, memory: MemoryHandler = None) -> str | None:
    user_input = user_input.lower()

    # Try to extract course code pattern (e.g., CSC 101, csc101)
    course_code_match = re.search(r'\b([a-z]{3})[\s\-]?(\d{3})\b', user_input)
    if course_code_match:
        code = course_code_match.group(1).upper() + " " + course_code_match.group(2)
        for dept, levels in course_data.items():
            for level, semesters in levels.items():
                for semester, courses in semesters.items():
                    for course in courses:
                        if course["code"].replace("-", " ").upper() == code:
                            if memory:
                                memory.update(department=dept, level=level)
                            return format_course_response(course, dept, level, semester)

    # Try to infer by department + level (e.g., "computer science 300 level")
    department, level = extract_department_and_level(user_input)
    if department or level:
        if memory:
            memory.update(department or memory.last_department, level or memory.last_level)

        dept = department or memory.last_department
        lvl = level or memory.last_level

        if dept and lvl:
            return list_all_courses(dept, lvl)

    return None  # Let fallback handle it


def format_course_response(course, dept, level, semester):
    return f"""
**📘 {course['code']} – {course['title']}**
- Department: `{dept.title()}`
- Level: `{level}`
- Semester: `{semester.title()}`
- Units: `{course['unit']}`
"""


def list_all_courses(department, level):
    try:
        semesters = course_data[department][level]
        lines = [f"📚 **{department.title()} – {level} Level Courses:**"]
        for semester, courses in semesters.items():
            lines.append(f"\n🟡 *{semester.title()} Semester*")
            for c in courses:
                lines.append(f"- `{c['code']}`: {c['title']} ({c['unit']} units)")
        return "\n".join(lines)
    except KeyError:
        return f"⚠️ Sorry, I couldn't find course details for `{department.title()} - {level} level`."


def extract_department_and_level(text):
    # Simplified department keywords (expand if needed)
    department_keywords = {
        "computer science": "computer science",
        "anatomy": "anatomy",
        "biochemistry": "biochemistry",
        "accounting": "accounting",
        "business admin": "business administration",
        "microbiology": "microbiology",
        "mass comm": "mass communication",
        "economics": "economics and operational research",
        "law": "law",
        "nursing": "nursing",
    }

    level_match = re.search(r"(\d{3})\s*(level)?", text)
    level = level_match.group(1) if level_match else None

    department = None
    for keyword, canonical in department_keywords.items():
        if keyword in text:
            department = canonical
            break

    return department, level
