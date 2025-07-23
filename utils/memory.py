# utils/memory.py

# Simple in-memory tracker for conversation context
user_memory = {
    "last_department": None,
    "last_level": None,
    "last_semester": None,
}

def update_memory(department=None, level=None, semester=None):
    if department:
        user_memory["last_department"] = department
    if level:
        user_memory["last_level"] = level
    if semester:
        user_memory["last_semester"] = semester

def get_last_department_level():
    return user_memory["last_department"], user_memory["last_level"], user_memory["last_semester"]

def clear_memory():
    user_memory["last_department"] = None
    user_memory["last_level"] = None
    user_memory["last_semester"] = None
