import streamlit as st
import json
import random
import re
import time
import openai

# Custom utility modules
from utils.embedding import load_qa_embeddings, find_most_similar_question
from utils.course_query import extract_course_info
from utils.preprocess import normalize_input
from utils.memory import MemoryHandler
from utils.greetings import detect_greeting, generate_greeting_response, is_small_talk, generate_small_talk_response
from utils.rewrite import rewrite_followup
from utils.search import fallback_to_gpt_if_needed

# Set up OpenAI API key
try:
    openai.api_key = st.secrets["OPENAI_API_KEY"]
except KeyError:
    st.error("❌ OpenAI API key not found in secrets. Please set OPENAI_API_KEY in .streamlit/secrets.toml")
    st.stop()

# Load data and model
qa_data, question_embeddings = load_qa_embeddings("data/crescent_qa.json")

# Initialize memory
memory = MemoryHandler()

# Confidence threshold
CONFIDENCE_THRESHOLD = 0.75


# 💬 Main chatbot function
def crescent_chatbot():
    st.set_page_config(page_title="Crescent University Chatbot", layout="centered")
    st.markdown("<h1 style='text-align: center;'>🎓 Crescent University Chatbot 🤖</h1>", unsafe_allow_html=True)

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # User input box
    user_input = st.chat_input("Ask me anything about Crescent University...")

    if user_input and user_input.strip():
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = get_bot_response(user_input)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})


# 🧠 Core logic for bot response
def get_bot_response(user_input):
    normalized_input = normalize_input(user_input)

    # Check greetings
    if detect_greeting(normalized_input):
        return generate_greeting_response()

    # Check small talk
    if is_small_talk(normalized_input):
        return generate_small_talk_response()

    # Rewrite follow-up using memory
    user_input = rewrite_followup(user_input, memory)

    # Try course-related query
    course_response = extract_course_info(user_input, memory)
    if course_response:
        return course_response

    # Semantic QA search
    best_match, score = find_most_similar_question(user_input, qa_data, question_embeddings)
    if score > CONFIDENCE_THRESHOLD:
        memory.update_last_topic(best_match)
        return best_match["answer"]

    # Fallback to GPT
    return fallback_to_gpt_if_needed(user_input)
