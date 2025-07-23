import streamlit as st
import time
import random
from utils.preprocess import normalize_input
from utils.embedding import load_embeddings
from utils.search import find_similar_question
from utils.memory import update_memory, get_last_department_level
from utils.rewrite import rewrite_query_with_memory
from utils.course_query import get_course_info
from utils.greetings import detect_greeting, generate_greeting_response
from utils.search import fallback_to_gpt_if_needed

# Load and apply CSS
def load_custom_css():
    with open("assets/style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_custom_css()


# Load embeddings
questions, answers, embeddings = load_embeddings("data/crescent_qa.json")

# Set up Streamlit page
st.set_page_config(page_title="Crescent University Chatbot", layout="wide")
st.markdown("<h2 style='text-align:center;'>🎓 Crescent University Chatbot</h2>", unsafe_allow_html=True)

# Typing animation
def bot_typing_animation():
    with st.spinner("Bot is typing..."):
        time.sleep(1.2)

# Initialize session memory
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Input box
user_input = st.chat_input("Ask me anything about Crescent University...")

# Message rendering
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle new message
if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Normalize and process input
    normalized_input = normalize_input(user_input)

    # Handle greetings
    if detect_greeting(normalized_input):
        bot_response = generate_greeting_response()
    else:
        # Rewrite based on memory (e.g., department, level)
        last_dept, last_level = get_last_department_level()
        rewritten_input = rewrite_query_with_memory(normalized_input, last_dept, last_level)

        # First try to get course info
        course_result = get_course_info(rewritten_input)
        if course_result:
            bot_response = course_result
        else:
            # Semantic search fallback
            best_q, best_a, score = find_similar_question(rewritten_input, questions, answers, embeddings)
            bot_response = fallback_to_gpt_if_needed(user_input, score, best_a)

    # Update memory if needed
    update_memory(normalized_input)

    # Bot typing
    bot_typing_animation()

    # Display bot response
    with st.chat_message("assistant"):
        st.markdown(bot_response)
    st.session_state.chat_history.append({"role": "assistant", "content": bot_response})
