# utils/rewrite.py

def rewrite_query_with_memory(query, last_department, last_level, last_semester):
    if last_department and "department" not in query.lower():
        query += f" in the {last_department} department"
    if last_level and "level" not in query.lower():
        query += f" at {last_level} level"
    if last_semester and "semester" not in query.lower():
        query += f" for the {last_semester} semester"
    return query.strip()
