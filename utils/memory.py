# utils/memory.py

def update_last_query_context(session_state, query_info):
    """
    Save the latest query context (department, level, semester) into Streamlit session state
    """
    if "last_query_info" not in session_state:
        session_state["last_query_info"] = {}

    if query_info.get("department"):
        session_state["last_query_info"]["department"] = query_info["department"]
        session_state["last_query_info"]["faculty"] = query_info.get("faculty")

    if query_info.get("level"):
        session_state["last_query_info"]["level"] = query_info["level"]

    if query_info.get("semester"):
        session_state["last_query_info"]["semester"] = query_info["semester"]


def get_last_query_context(session_state):
    """
    Retrieve the last query context (department, level, semester) from memory
    """
    return session_state.get("last_query_info", {})
