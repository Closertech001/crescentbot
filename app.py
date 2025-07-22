# app.py

import streamlit as st
from utils.embedding import load_model_and_embeddings
from utils.search import get_best_match
from utils.memory import MemoryHandler
from utils.greetings import detect_greeting, generate_greeting_response
from utils.rewrite import rewrite_followup
from utils.course_query import extract_course_info
import os
import openai

# 🔐 OpenAI Key
openai.api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", "")

# 🚀 Page Setup
st.set_page_config(page_title="🎓 Crescent University Chatbot", layout="centered")

# 🧠 Session state
if "messages" not in st.session_state:
    st.session_state.messages = []

memory = MemoryHandler()

# 🔍 Load embeddings & model
model, data, index, embeddings = load_model_and_embeddings()

# 🎓 Header
st.title("🎓 Crescent University Chatbot")
st.markdown("I'm here to help with your questions about **courses, departments, and requirements** at Crescent University.")

# 💬 Show chat history
with st.container():
    for msg in st.session_state.messages:
        avatar = "🧑‍🎓" if msg["role"] == "user" else "🤖"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])

# 📥 User input
user_input = st.chat_input("Type your question here...")

if user_input:
    # Save user message
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Handle greeting
    if detect_greeting(user_input):
        response = generate_greeting_response()

    # Handle course-related questions
    elif any(word in user_input.lower() for word in ["course", "unit", "semester", "level", "department", "faculty"]):
        response = extract_course_info(user_input, memory)
        if not response or "provide department" in response.lower():
            response = "⚠️ Please provide the **department**, **level**, and **semester** so I can find the correct course information."

    # Semantic Q&A matching
    else:
        match = get_best_match(user_input, model, index, data, embeddings)
        response = match if match else "🤔 I'm not sure how to answer that. Can you rephrase or provide more details?"

    # Optional tone rewrite
    response = rewrite_followup(response, memory)

    # Save bot response
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant", avatar="🤖"):
        st.markdown(response)
