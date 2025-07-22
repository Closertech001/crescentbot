# utils/memory.py

# Memory tracking for context
class Memory:
    def __init__(self):
        self.last_department = None
        self.last_level = None

    def update(self, department=None, level=None):
        if department:
            self.last_department = department
        if level:
            self.last_level = level

    def get(self):
        return self.last_department, self.last_level
