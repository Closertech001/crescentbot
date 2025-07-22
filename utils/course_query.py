# utils/course_query.py

import json
import re

def extract_course_info(question, memory, course_data_path="data/course_data.json"):
    dept = memory.get("department")
    level = memory.get("level")
    semester = memory.get("semester")

    # Try to extract department, level, and semester if not already in memory
    question_lower = question.lower()

    # Simple rule-based extraction
    if not dept:
        match = re.search(r"(computer science|anatomy|biochemistry|accounting|business administration|microbiology|economics and operational research|mass communication|law|nursing)", question_lower)
        if match:
            dept = match.group(1)
            memory.set("department", dept)

    if not level:
        match = re.search(r"(\d{3}) level", question_lower)
        if match:
            level = match.group(1)
            memory.set("level", level)

    if not semester:
        if "first semester" in question_lower:
            semester = "first"
            memory.set("semester", semester)
        elif "second semester" in question_lower:
            semester = "second"
            memory.set("semester", semester)

    # Validate all info is present
    if not all([dept, level, semester]):
        return "Please provide department, level, and semester to get course information."

    try:
        with open(course_data_path, "r", encoding="utf-8") as f:
            course_data = json.load(f)
    except Exception as e:
        return f"Error loading course data: {e}"

    # Search course info
    dept_data = course_data.get(dept.lower())
    if not dept_data:
        return f"No data found for department: {dept}"

    level_data = dept_data.get(level)
    if not level_data:
        return f"No data found for {dept} at {level} level."

    semester_data = level_data.get(semester.lower())
    if not semester_data:
        return f"No course info for {dept} {level} level {semester} semester."

    # Build response
    lines = [f"📘 **{dept.title()} {level} Level - {semester.title()} Semester Courses:**"]
    for course in semester_data:
        code = course.get("code", "N/A")
        title = course.get("title", "Untitled")
        unit = course.get("unit", "N/A")
        lines.append(f"- `{code}`: **{title}** ({unit} unit{'s' if unit != '1' else ''})")

    return "\n".join(lines)
