import streamlit as st
from typing import Dict, Any


def store_context_from_query(query_info: Dict[str, Any]) -> None:
    """
    Store department and level from the current query into session state.

    Args:
        query_info (dict): Dictionary containing query information with optional 'department' and 'level'.
    """
    if not query_info:
        return

    if query_info.get("department"):
        st.session_state["last_department"] = query_info["department"]

    if query_info.get("level"):
        st.session_state["last_level"] = query_info["level"]


def enrich_query_with_context(query_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Enrich the current query with department/level from session state if missing.

    Args:
        query_info (dict): The current query dictionary to be enriched.

    Returns:
        dict: Enriched query_info with department and level filled in if available.
    """
    if not query_info.get("department") and "last_department" in st.session_state:
        query_info["department"] = st.session_state["last_department"]

    if not query_info.get("level") and "last_level" in st.session_state:
        query_info["level"] = st.session_state["last_level"]

    return query_info
