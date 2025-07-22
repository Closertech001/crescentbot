# utils/memory.py

import datetime

# utils/memory.py

def update_memory(memory, key, value):
    """
    Updates the in-session memory dictionary with a new key-value pair.
    """
    memory[key] = value
    return memory

def get_last_context(memory, key):
    """
    Retrieves the last stored value for a given key from memory.
    """
    return memory.get(key, None)
