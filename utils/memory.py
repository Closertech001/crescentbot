# utils/memory.py

from typing import Dict, Any

class Memory:
    def __init__(self):
        self._state = {}

    def update(self, new_info: Dict[str, Any]):
        self._state.update(new_info)

    def get_memory(self) -> Dict[str, Any]:
        return self._state
