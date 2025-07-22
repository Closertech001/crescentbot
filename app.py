# app.py

import streamlit as st
import time
import random
import re
import openai
from utils.preprocess import preprocess_input
from utils.course_query import get_course_info
from utils.memory import update_memory, get_last_context
from utils.embedding import load_qa_embeddings
from utils.search import semantic_search
from utils.rewrite import rewrite_followup
from utils.greetings import is_greeting, get_greeting_response
from utils.log_utils import log_interaction

# Load secrets
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Load embedded QA data
qa_data, embeddings, model = load_qa_embeddings()

# Streamlit UI setup
st.set_page_config(page_title="Crescent University Chatbot", layout="centered")
st.markdown("<h2 style='text-align:center;'>🎓 Crescent University Chatbot 🤖</h2>", unsafe_allow_html=True)

# Session state for conversation
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "memory" not in st.session_state:
    st.session_state.memory = {}

# Input box
user_input = st.text_input("Ask me anything about Crescent University...", key="input", placeholder="e.g., What are the 100 level courses in Computer Science?")

# Typing animation
def bot_typing():
    with st.empty():
        for dots in ["", ".", "..", "..."]:
            st.markdown(f"**Bot is typing{dots}**")
            time.sleep(0.4)

# Helper: display messages
def display_chat():
    for role, message in st.session_state.chat_history:
        align = "left" if role == "user" else "right"
        bubble_color = "#f0f0f0" if role == "user" else "#d3f9d8"
        st.markdown(
            f"""
            <div style="text-align: {align}; background-color: {bubble_color}; padding: 10px; border-radius: 12px; margin: 5px;">
                <b>{'You' if role == 'user' else 'Bot'}:</b> {message}
            </div>
            """,
            unsafe_allow_html=True,
        )

# Handle interaction
def handle_query(query):
    preprocessed = preprocess_input(query)
    norm_text = preprocessed["normalized"]
    dept_context = get_last_context(st.session_state.memory, "department")

    # Greeting detection
    if is_greeting(query):
        reply = get_greeting_response()
        st.session_state.chat_history.append(("bot", reply))
        return

    # Small talk / fallback
    small_talk_phrases = [
        "Here's what I found for you:",
        "Let me help with that.",
        "Sure! Here’s the info:",
        "Absolutely! Here’s something useful:",
        "Of course. This might help:"
    ]

    # Try course-based query
    course_response = get_course_info(norm_text, memory=st.session_state.memory)
    if course_response:
        st.session_state.chat_history.append(("bot", random.choice(small_talk_phrases) + " " + course_response))
        return

    # Try semantic search
    top_match, score = semantic_search(norm_text, model, embeddings, qa_data)
    if score > 0.7:
        st.session_state.chat_history.append(("bot", random.choice(small_talk_phrases) + " " + top_match["answer"]))
        return

    # Rewrite for GPT if low similarity
    if dept_context:
        norm_text = rewrite_followup(norm_text, dept_context)

    bot_typing()
    try:
        gpt_response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant for students of Crescent University."},
                {"role": "user", "content": norm_text}
            ],
            temperature=0.4,
            max_tokens=400
        )
        answer = gpt_response["choices"][0]["message"]["content"]
        st.session_state.chat_history.append(("bot", answer))
    except Exception as e:
        st.session_state.chat_history.append(("bot", "Sorry, I couldn’t fetch an answer at the moment."))

    # Log interaction
    log_interaction(query, answer)

# Run chatbot
if user_input:
    st.session_state.chat_history.append(("user", user_input))
    handle_query(user_input)
    st.session_state.input = ""  # Clear input field

display_chat()

# Clear chat button
if st.button("Clear Chat"):
    st.session_state.chat_history = []
    st.session_state.memory = {}
    st.experimental_rerun()
