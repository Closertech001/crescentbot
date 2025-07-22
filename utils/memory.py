from datetime import datetime

def update_memory(memory: dict, key: str, value, with_timestamp: bool = False) -> dict:
    """
    Updates the in-session memory dictionary with a key-value pair.
    
    Args:
        memory (dict): The memory dictionary to update.
        key (str): The key to store the value under.
        value (Any): The value to store.
        with_timestamp (bool): Whether to store a timestamp with the value.

    Returns:
        dict: Updated memory dictionary.
    """
    if with_timestamp:
        memory[key] = {
            "value": value,
            "timestamp": datetime.utcnow().isoformat()
        }
    else:
        memory[key] = value
    return memory

def get_last_context(memory: dict, key: str):
    """
    Retrieves the last stored value for a given key from memory.
    
    Args:
        memory (dict): The memory dictionary.
        key (str): The key to look up.

    Returns:
        Any: The stored value, or None if key is not found.
    """
    entry = memory.get(key, None)
    if isinstance(entry, dict) and "value" in entry:
        return entry["value"]
    return entry
