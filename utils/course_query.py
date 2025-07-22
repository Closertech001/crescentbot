# utils/course_query.py

import json
import re
from difflib import SequenceMatcher
from utils.memory import MemoryHandler

with open("data/course_data.json", "r", encoding="utf-8") as f:
    COURSE_DATA = json.load(f)

def clean_text(text):
    return re.sub(r"[^a-zA-Z0-9\s]", "", text).strip().lower()

def similarity(a, b):
    return SequenceMatcher(None, clean_text(a), clean_text(b)).ratio()

def extract_course_info(user_input: str, memory: MemoryHandler) -> str | None:
    user_input = user_input.lower()

    dept = memory.get("department")
    level = memory.get("level")
    semester = memory.get("semester")

    # Try to extract department
    for department in COURSE_DATA:
        if department.lower() in user_input:
            dept = department
            memory.set("department", dept)

    # Try to extract level
    for l in ["100", "200", "300", "400"]:
        if l in user_input:
            level = f"{l} level"
            memory.set("level", level)

    # Try to extract semester
    if "first semester" in user_input:
        semester = "first"
        memory.set("semester", semester)
    elif "second semester" in user_input:
        semester = "second"
        memory.set("semester", semester)

    # If department, level, and semester are present, return full list
    if dept and level and semester:
        try:
            courses = COURSE_DATA[dept][level][semester]
            course_lines = [f"- `{c['code']}`: {c['title']} ({c['unit']} units)" for c in courses]
            return f"Here are the courses for **{dept.title()} - {level.title()} - {semester.title()} Semester**:\n\n" + "\n".join(course_lines)
        except KeyError:
            return f"Sorry, I couldn't find courses for {dept} {level} {semester}."

    # Try course-specific match
    all_courses = []
    for dpt, lvls in COURSE_DATA.items():
        for lvl, sems in lvls.items():
            for sem, course_list in sems.items():
                for course in course_list:
                    all_courses.append((dpt, lvl, sem, course))

    user_input_clean = clean_text(user_input)
    best_match = max(all_courses, key=lambda x: max(
        similarity(user_input_clean, x[3]['title']),
        similarity(user_input_clean, x[3]['code'])
    ))

    match_score = max(
        similarity(user_input_clean, best_match[3]['title']),
        similarity(user_input_clean, best_match[3]['code'])
    )

    if match_score > 0.7:
        course = best_match[3]
        return f"**{course['code']}** — {course['title']} ({course['unit']} units) under {best_match[0]} ({best_match[1]}, {best_match[2]} semester)."

    return None
