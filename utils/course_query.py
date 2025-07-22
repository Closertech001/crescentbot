import json
import re
from rapidfuzz import fuzz, process

class CourseQueryHandler:
    def __init__(self, course_data_path):
        with open(course_data_path, "r", encoding="utf-8") as f:
            self.course_data = json.load(f)

        self.dept_to_college = {
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

        self.dept_aliases = {
            "comp sci": "computer science",
            "mass comm": "mass communication",
            "biz admin": "business administration",
            "med lab": "medical laboratory science",
        }

    def normalize_department(self, name):
        name = name.lower().strip()
        if name in self.dept_aliases:
            return self.dept_aliases[name]
        best_match, score, _ = process.extractOne(name, list(self.dept_to_college.keys()), scorer=fuzz.ratio)
        return best_match if score > 70 else None

    def extract_info(self, user_input):
        user_input = user_input.lower()

        # Extract level
        level_match = re.search(r"\b(100|200|300|400|500)\s*(?:level)?\b", user_input)
        level = level_match.group(1) if level_match else None

        # Extract semester
        semester_match = re.search(r"\b(first|1st|second|2nd)\s+semester\b", user_input)
        semester = (
            "first" if semester_match and "1" in semester_match.group(1)
            else "second" if semester_match
            else None
        )

        # Extract department name
        dept_match = re.search(r"\b(?:course[s]?|subject[s]?)\s+(?:for|in|of)\s+([a-zA-Z &]+)", user_input)
        dept_name = dept_match.group(1).strip() if dept_match else None

        # Fallback matching
        if not dept_name:
            for alias, actual in self.dept_aliases.items():
                if alias in user_input:
                    dept_name = actual
                    break
            for dept in self.dept_to_college:
                if dept in user_input:
                    dept_name = dept
                    break

        normalized_dept = self.normalize_department(dept_name or "")
        if not normalized_dept:
            return "❌ I couldn't identify the department. Please try again using the full or common name."

        college = self.dept_to_college[normalized_dept]
        courses = self.course_data.get(college, {}).get(normalized_dept, {}).get("courses", {})

        # Filter courses
        filtered = []
        for course in courses:
            if level and str(course.get("level")) != str(level):
                continue
            if semester and course.get("semester", "").lower() != semester:
                continue
            filtered.append(course)

        if not filtered:
            available_levels = sorted({c["level"] for c in courses})
            available_semesters = sorted({c.get("semester", "").lower() for c in courses})
            return (
                f"No courses found for **{normalized_dept.title()}** "
                f"{f'{level}-level' if level else ''} {f'{semester} semester' if semester else ''}.\n\n"
                f"✅ Available levels: {', '.join(map(str, available_levels))}\n"
                f"✅ Available semesters: {', '.join(available_semesters)}"
            )

        # Build response
        response = f"📚 **Courses for {normalized_dept.title()}**"
        if level:
            response += f" - {level}-level"
        if semester:
            response += f" ({semester.title()} Semester)"
        response += ":\n\n"

        for course in filtered:
            code = course.get("code", "N/A")
            title = course.get("title", "Untitled")
            unit = course.get("unit", "N/A")
            response += f"- `{code}`: {title} ({unit} units)\n"

        return response
