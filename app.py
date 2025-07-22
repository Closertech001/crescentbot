import streamlit as st
import random
import time
import json
import os
import re
from openai import OpenAI
from utils.course_query import extract_course_info, get_course_by_code
from utils.embedding import load_model, get_question_embeddings
from utils.search import get_top_k_matches as semantic_search
from utils.memory import update_memory, get_last_context
from utils.preprocess import normalize_text
from utils.rewrite import rewrite_with_tone

# Load data and model once
@st.cache_resource
def load():
    with open("data/crescent_qa.json", "r", encoding="utf-8") as f:
        qa_data = json.load(f)
    questions = [q["question"] for q in qa_data]
    model = load_model("all-MiniLM-L6-v2")
    embeddings = get_question_embeddings(questions, model)
    return qa_data, model, embeddings

qa_data, model, question_embeddings = load()

# OpenAI setup
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# UI Config
st.set_page_config(page_title="Crescent UniBot 🤖", layout="wide")
st.markdown("<h3 style='text-align:center;'>🤖 Crescent University Chatbot</h3>", unsafe_allow_html=True)

# Typing animation
def typing_animation():
    with st.empty():
        for dots in ["", ".", "..", "..."]:
            st.markdown(f"**Bot is typing{dots}**")
            time.sleep(0.3)

# Inline Greeting + Farewell Detection
def detect_greeting(text):
    greetings = ["hi", "hello", "hey", "good morning", "good evening", "good afternoon"]
    return any(greet in text.lower() for greet in greetings)

def detect_farewell(text):
    farewells = ["bye", "goodbye", "see you", "farewell", "later"]
    return any(farewell in text.lower() for farewell in farewells)

def get_random_greeting():
    return random.choice([
        "Hello there! 👋",
        "Hi! How can I help you today?",
        "Hey! I'm here to assist you.",
        "Welcome! Ask me anything about Crescent University.",
        "Hiya! 😊 What do you want to know?"
    ])

def get_random_farewell():
    return random.choice([
        "Goodbye! 👋",
        "See you later!",
        "Take care!",
        "Bye for now!",
        "Wishing you all the best!"
    ])

# Handle Input
def handle_input(user_input):
    user_input_norm = normalize_text(user_input)
    update_memory(user_input_norm)

    # Greeting or farewell
    if detect_greeting(user_input):
        return get_random_greeting()
    if detect_farewell(user_input):
        return get_random_farewell()

    # Course code lookup
    course_code_match = get_course_by_code(user_input_norm)
    if course_code_match:
        return course_code_match

    # Department/course-related info
    course_response = extract_course_info(user_input_norm)
    if course_response:
        return course_response

    # Semantic search fallback
    matched = semantic_search(user_input_norm, qa_data, question_embeddings, model)
    if matched:
        return rewrite_with_tone(user_input, matched)

    # GPT-4 fallback
    try:
        gpt_response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant for Crescent University."},
                {"role": "user", "content": user_input}
            ],
            temperature=0.7
        )
        return rewrite_with_tone(user_input, gpt_response.choices[0].message.content)
    except Exception as e:
        return "Sorry, I encountered an error while processing your request."

# Chat loop
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.chat_input("Ask me about Crescent University...")

if user_input:
    st.session_state.chat_history.append(("user", user_input))
    typing_animation()
    response = handle_input(user_input)
    st.session_state.chat_history.append(("bot", response))

for role, message in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(message)
