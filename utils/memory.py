# utils/memory.py

class Memory:
    """
    Tracks conversational memory like department and level.
    """
    def __init__(self):
        self.context = {
            "department": None,
            "level": None,
            "semester": None
        }

    def update(self, **kwargs):
        """
        Update memory context. Accepts any of: department, level, semester.
        """
        for key in kwargs:
            if key in self.context:
                self.context[key] = kwargs[key]

    def get(self):
        """
        Returns current memory values as a dict.
        """
        return self.context.copy()

    def reset(self):
        """
        Resets the memory to default state.
        """
        for key in self.context:
            self.context[key] = None

    def __str__(self):
        return f"Memory({self.context})"
