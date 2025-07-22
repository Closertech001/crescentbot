# utils/memory.py

class MemoryHandler:
    def __init__(self):
        self.last_department = None
        self.last_level = None
        self.last_topic = None

    def update(self, department=None, level=None):
        if department:
            self.last_department = department
        if level:
            self.last_level = level

    def update_last_topic(self, topic):
        self.last_topic = topic

    def get(self):
        return self.last_department, self.last_level, self.last_topic
