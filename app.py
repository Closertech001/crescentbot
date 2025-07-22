# app.py

import streamlit as st
import random
import time
import json
import re
from utils.course_query import get_course_info
from utils.embedding import load_qa_data, get_top_k_matches
from utils.rewrite import rewrite_with_tone
from utils.preprocess import normalize_input
from utils.memory import update_memory, get_last_context
from openai import OpenAI
import os

# =============== INLINE GREETING UTILS ===============

def detect_greeting(text):
    greetings = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"]
    return any(greet in text.lower() for greet in greetings)

def detect_farewell(text):
    farewells = ["bye", "goodbye", "see you", "later", "take care"]
    return any(farewell in text.lower() for farewell in farewells)

def get_random_greeting():
    return random.choice([
        "Hi there! 😊 How can I assist you today?",
        "Hello! 👋 What would you like to know?",
        "Hey! 👨‍🎓 Ready to explore Crescent University info?",
        "Welcome! 🤝 Ask me anything about CUAB.",
        "Hi! 🚀 I’m your Crescent Uni assistant. How can I help?"
    ])

# =============== INLINE TONE DETECTION ===============

def detect_tone(text):
    text = text.lower()
    if any(word in text for word in ["pls", "please", "hi", "hello", "thank", "good morning", "good afternoon", "good evening"]):
        return "polite"
    if any(word in text for word in ["urgent", "now", "quick", "fast", "immediately", "asap"]):
        return "urgent"
    if any(word in text for word in ["why", "what", "how", "when", "confused", "help", "explain", "not sure", "don't understand"]):
        return "confused"
    if any(word in text for word in ["angry", "nonsense", "rubbish", "dumb", "idiot", "annoyed", "useless", "frustrated", "mad"]):
        return "angry"
    if re.search(r"[!?]{2,}", text):
        return "emphatic"
    return "neutral"

# =============== INITIALIZE ===============

st.set_page_config(page_title="Crescent University Chatbot", page_icon="🎓", layout="centered")
st.markdown("<h2 style='text-align:center;'>🎓 Crescent University Chatbot</h2>", unsafe_allow_html=True)

# Load QA data + embeddings
qa_data, question_embeddings = load_qa_data()

# Load OpenAI API
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Session state init
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "memory" not in st.session_state:
    st.session_state.memory = {}

# =============== INPUT HANDLER ===============

def handle_input(user_input):
    norm_input = normalize_input(user_input)

    # Greetings
    if detect_greeting(norm_input):
        return get_random_greeting()

    if detect_farewell(norm_input):
        return random.choice(["Goodbye! 👋", "See you soon!", "Take care!", "Bye for now!", "Later! 🚀"])

    # Update memory
    update_memory(norm_input, st.session_state.memory)

    # 1. Try course info
    course_response = get_course_info(norm_input, st.session_state.memory)
    if course_response:
        return rewrite_with_tone(user_input, course_response)

    # 2. Semantic match
    matches = get_top_k_matches(norm_input, qa_data, question_embeddings, k=3)
    if matches:
        best = matches[0]
        return rewrite_with_tone(user_input, best["answer"])

    # 3. Fallback to GPT
    gpt_response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You're an assistant for Crescent University. Answer clearly and helpfully."},
            {"role": "user", "content": norm_input}
        ],
        temperature=0.4
    )
    gpt_answer = gpt_response.choices[0].message.content
    return rewrite_with_tone(user_input, gpt_answer)

# =============== INPUT SUBMIT LOGIC ===============

def submit():
    st.session_state.submitted = True
    st.session_state.last_input = st.session_state.user_input
    st.session_state.user_input = ""  # Safe clear

# =============== CHAT UI ===============

st.text_input("You:", key="user_input", on_change=submit, placeholder="Ask me about courses, departments, admission...")

# Typing + Chat loop
if st.session_state.get("submitted"):
    user_input = st.session_state.get("last_input", "")
    with st.spinner("Bot is typing..."):
        time.sleep(1.3)
        response = handle_input(user_input)

    st.session_state.chat_history.append(("You", user_input))
    st.session_state.chat_history.append(("Bot", response))
    st.session_state.submitted = False

# =============== RENDER CHAT HISTORY ===============
for sender, msg in st.session_state.chat_history:
    if sender == "You":
        st.markdown(f"<div style='text-align:right; color:#0000cc;'>**{sender}:** {msg}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div style='text-align:left; color:#008000;'>**{sender}:** {msg}</div>", unsafe_allow_html=True)
