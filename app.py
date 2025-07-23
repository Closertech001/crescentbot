import streamlit as st
from utils.preprocess import normalize_input
from utils.embedding import load_embeddings
from utils.search import find_similar_question
from utils.memory import update_memory, get_last_department_level
from utils.course_query import get_course_info
from utils.greetings import detect_greeting, get_dynamic_greeting
from utils.rewrite import rewrite_query_with_memory
from utils.search import fallback_to_gpt_if_needed
import time

# Load the embeddings once
embedding_data = load_embeddings()

# Initialize Streamlit
st.set_page_config(page_title="Crescent University Chatbot", page_icon="🎓")
st.markdown("<h2 style='text-align: center;'>🎓 Crescent University Chatbot</h2>", unsafe_allow_html=True)

# --- Session State ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "last_department" not in st.session_state:
    st.session_state.last_department = None

if "last_level" not in st.session_state:
    st.session_state.last_level = None

if "last_semester" not in st.session_state:
    st.session_state.last_semester = None

# --- Chat Input ---
user_input = st.chat_input("Ask me anything about Crescent University...")

# --- Handle Input ---
if user_input:
    # Add user message
    st.session_state.chat_history.append(("user", user_input))

    # Typing indicator
    with st.empty():
        for dots in [".", "..", "..."]:
            st.markdown(f"**Bot is typing{dots}**")
            time.sleep(0.3)

    # Normalize
    cleaned_input = normalize_input(user_input)

    # Check if greeting/small talk
    if detect_greeting(cleaned_input):
        bot_response = get_dynamic_greeting()
    else:
        # Rewrite query using memory
        enriched_query = rewrite_query_with_memory(
            cleaned_input,
            st.session_state.last_department,
            st.session_state.last_level,
            st.session_state.last_semester
        )

        # Check for course info first (code/department/level/semester)
        course_result = get_course_info(enriched_query)

        if course_result:
            bot_response = course_result
        else:
            # Fallback to embedding search
            matched_question, matched_answer, score = find_similar_question(enriched_query, embedding_data)

            bot_response = fallback_to_gpt_if_needed(
                user_input=cleaned_input,
                similarity_score=score,
                matched_answer=matched_answer
            )

    # Update memory
    update_memory(user_input, st.session_state)

    # Add bot response
    st.session_state.chat_history.append(("bot", bot_response))

# --- Display Chat History ---
for sender, message in st.session_state.chat_history:
    if sender == "user":
        st.chat_message("user").write(message)
    else:
        st.chat_message("assistant").write(message)
