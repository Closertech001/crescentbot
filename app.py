import streamlit as st
from utils.preprocess import normalize_input
from utils.embedding import load_embeddings
from utils.search import find_similar_question, fallback_to_gpt_if_needed
from utils.memory import update_memory, get_last_department_level
from utils.course_query import get_course_info
from utils.greetings import detect_greeting, get_dynamic_greeting
from utils.rewrite import rewrite_query_with_memory
import time

# Load embeddings from data/crescent_qa.json
embedding_data = load_embeddings("data/crescent_qa.json")

# Streamlit app config
st.set_page_config(page_title="Crescent University Chatbot", page_icon="🎓")
st.markdown("<h2 style='text-align: center;'>🎓 Crescent University Chatbot</h2>", unsafe_allow_html=True)

# Session State Initialization
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "last_department" not in st.session_state:
    st.session_state.last_department = None

if "last_level" not in st.session_state:
    st.session_state.last_level = None

if "last_semester" not in st.session_state:
    st.session_state.last_semester = None

# Chat Input
user_input = st.chat_input("Ask me anything about Crescent University...")

if user_input:
    # Show user message
    st.session_state.chat_history.append(("user", user_input))

    # Show typing animation
    with st.empty():
        for dots in [".", "..", "..."]:
            st.markdown(f"**Bot is typing{dots}**")
            time.sleep(0.3)

    # Normalize input
    cleaned_input = normalize_input(user_input)

    # Greeting/Small talk check
    if detect_greeting(cleaned_input):
        bot_response = get_dynamic_greeting()
    else:
        # Use memory to enrich context
        enriched_query = rewrite_query_with_memory(
            cleaned_input,
            st.session_state.last_department,
            st.session_state.last_level,
            st.session_state.last_semester
        )

        # First, try getting structured course info
        course_result = get_course_info(enriched_query, "data/course_data.json")

        if course_result:
            bot_response = course_result
        else:
            # Otherwise, use semantic Q&A search
            matched_question, matched_answer, score = find_similar_question(enriched_query, embedding_data)

            # GPT fallback if needed
            bot_response = fallback_to_gpt_if_needed(
                user_input=cleaned_input,
                similarity_score=score,
                matched_answer=matched_answer
            )

    # Update session memory
    update_memory(user_input, st.session_state)

    # Save bot response
    st.session_state.chat_history.append(("bot", bot_response))

# Display full chat history
for sender, message in st.session_state.chat_history:
    if sender == "user":
        st.chat_message("user").write(message)
    else:
        st.chat_message("assistant").write(message)
