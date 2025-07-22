def store_context_from_query(query_info):
    if not query_info:
        return
    if "department" in query_info and query_info["department"]:
        import streamlit as st
        st.session_state["last_department"] = query_info["department"]
    if "level" in query_info and query_info["level"]:
        st.session_state["last_level"] = query_info["level"]

def enrich_query_with_context(query_info):
    import streamlit as st
    if not query_info.get("department") and "last_department" in st.session_state:
        query_info["department"] = st.session_state["last_department"]
    if not query_info.get("level") and "last_level" in st.session_state:
        query_info["level"] = st.session_state["last_level"]
    return query_info
