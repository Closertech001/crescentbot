import streamlit as st
import json
import random
import re
import time
from utils.embedding import load_qa_embeddings, find_most_similar_question
from utils.course_query import extract_course_info
from utils.preprocess import normalize_input
from utils.memory import MemoryHandler
from utils.greetings import detect_greeting, generate_greeting_response, is_small_talk, generate_small_talk_response
from utils.rewrite import rewrite_followup
from utils.search import fallback_to_gpt_if_needed
import openai

# --- Load OpenAI API key ---
openai.api_key = st.secrets["OPENAI_API_KEY"]

# --- Initialize memory handler ---
memory = MemoryHandler()

# --- Load question-answer data and embeddings ---
qa_data, question_embeddings = load_qa_embeddings("data/crescent_qa.json")

# --- Streamlit UI Setup ---
st.set_page_config(page_title="Crescent University Chatbot", layout="centered")
st.markdown("<h1 style='text-align: center;'>🎓 Crescent University Chatbot 🤖</h1>", unsafe_allow_html=True)
st.markdown("<style>div.stTextInput>div>input { font-size: 18px; }</style>", unsafe_allow_html=True)

# --- Initialize chat history ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Chatbot Response Function ---
def get_bot_response(user_input):
    normalized_input = normalize_input(user_input)

    # Check for greetings or small talk
    if detect_greeting(normalized_input):
        return generate_greeting_response()
    if is_small_talk(normalized_input):
        return generate_small_talk_response()

    # Use memory for follow-ups
    user_input = rewrite_followup(user_input, memory)

    # Try extracting course info
    course_response = extract_course_info(user_input, memory)
    if course_response:
        return course_response

    # Semantic search in QA pairs
    best_match, score = find_most_similar_question(user_input, qa_data, question_embeddings)
    if score > 0.75:
        memory.update_last_topic(best_match)
        return best_match["answer"]

    # Fallback to GPT for unmatched questions
    return fallback_to_gpt_if_needed(user_input)

# --- Chat Input and Display ---
st.markdown("---")
user_input = st.text_input("You:", "", key="input")
submit = st.button("Send")

if submit and user_input:
    st.session_state.messages.append(("user", user_input))
    with st.spinner("Bot is typing..."):
        time.sleep(random.uniform(0.4, 1.2))
        bot_response = get_bot_response(user_input)
        st.session_state.messages.append(("bot", bot_response))
        st.experimental_rerun()

# --- Display chat history ---
for role, message in st.session_state.messages:
    if role == "user":
        st.markdown(f"<div style='text-align: right; color: #1a73e8;'><b>You:</b> {message}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div style='text-align: left; color: #34a853;'><b>Bot:</b> {message}</div>", unsafe_allow_html=True)
