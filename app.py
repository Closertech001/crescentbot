import streamlit as st
import json
import random
import time
import re
import torch
from sentence_transformers import SentenceTransformer
from utils.course_query import extract_course_info, get_course_by_code
from utils.preprocess import normalize_input
from utils.embedding import get_question_embeddings, load_model
from utils.search import semantic_search
from utils.greetings import (
    detect_greeting, detect_farewell, is_small_talk,
    get_random_greeting, rewrite_with_tone
)
from utils.memory import update_memory, get_contextual_input

# ----- Streamlit Config -----
st.set_page_config(page_title="Crescent University Chatbot", layout="centered")

# ----- Typing Animation -----
def typing_animation(text, delay=0.03):
    message_placeholder = st.empty()
    displayed = ""
    for char in text:
        displayed += char
        message_placeholder.markdown(displayed)
        time.sleep(delay)

# ----- Session Memory -----
if "memory" not in st.session_state:
    st.session_state.memory = {"department": None, "level": None, "semester": None}

# ----- Load Data -----
@st.cache_data
def load_course_data():
    with open("data/course_data.json", "r") as f:
        return json.load(f)

@st.cache_data
def load_qa_data():
    with open("data/crescent_qa.json", "r") as f:
        return json.load(f)

qa_data = load_qa_data()
questions = [item["question"] for item in qa_data]
answers = [item["answer"] for item in qa_data]
model = load_model()
question_embeddings = get_question_embeddings(questions)

course_data = load_course_data()

# ----- Main Logic -----
def handle_input(raw_input):
    user_input = normalize_input(raw_input)
    memory = st.session_state.memory

    # Greeting & small talk
    if detect_greeting(user_input):
        return get_random_greeting()
    if detect_farewell(user_input):
        return "Bye for now! 😊 Let me know if you need anything else."
    if is_small_talk(user_input):
        return random.choice([
            "All good here! Ready to assist.",
            "Great! Ask me anything about Crescent University.",
            "Feeling sharp! How can I help?"
        ])

    # Course code e.g. "What is MTH 101?"
    course_info = get_course_by_code(user_input, course_data)
    if course_info:
        return rewrite_with_tone(user_input, course_info)

    # Direct course extraction e.g. “Courses for 200 level Computer Science”
    course_response = extract_course_info(user_input, course_data, memory)
    if course_response:
        return rewrite_with_tone(user_input, course_response)

    # Fallback to GPT-style QA from crescent_qa.json
    updated_input = get_contextual_input(user_input, memory)
    top_questions, top_idx = semantic_search(updated_input, questions, question_embeddings, model)
    if top_idx:
        update_memory(user_input, memory)
        return rewrite_with_tone(user_input, answers[top_idx[0]])

    return "Sorry, I couldn’t find anything helpful. Could you rephrase or be more specific?"

# ----- UI -----
st.markdown("<h2 style='text-align: center;'>🎓 Crescent University Chatbot</h2>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

user_input = st.text_input("Ask your question here...", key="user_input_box")

if user_input:
    with st.spinner("Bot is typing..."):
        response = handle_input(user_input)
        typing_animation(response)
    st.session_state.user_input_box = ""
