import streamlit as st
import uuid

def init_memory(reset=False):
    if reset or "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if reset or "related_questions" not in st.session_state:
        st.session_state.related_questions = []
    if reset or "last_department" not in st.session_state:
        st.session_state.last_department = None
