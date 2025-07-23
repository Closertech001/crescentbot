# Memory handler using Streamlit session_state
import streamlit as st

def get_last_query_info():
    return st.session_state.get("last_query_info", {})

def set_last_query_info(info):
    st.session_state["last_query_info"] = info
