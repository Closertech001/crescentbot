# utils/memory.py

class MemoryHandler:
    def __init__(self):
        self.memory = {}

    def get(self, key, default=None):
        return self.memory.get(key, default)

    def set(self, key, value):
        self.memory[key] = value

    def clear(self):
        self.memory.clear()
