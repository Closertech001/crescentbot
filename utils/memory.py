import streamlit as st

def init_memory():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "related_questions" not in st.session_state:
        st.session_state.related_questions = []
    if "last_department" not in st.session_state:
        st.session_state.last_department = None
