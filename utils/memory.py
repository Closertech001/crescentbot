# utils/memory.py

memory_store = {
    "last_department": None,
    "last_level": None,
    "last_semester": None,
}

def update_memory(department=None, level=None, semester=None):
    if department:
        memory_store["last_department"] = department
    if level:
        memory_store["last_level"] = level
    if semester:
        memory_store["last_semester"] = semester

def get_last_department_level():
    return (
        memory_store.get("last_department"),
        memory_store.get("last_level"),
        memory_store.get("last_semester"),
    )
