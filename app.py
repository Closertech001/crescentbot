# app.py - Crescent University Chatbot

import streamlit as st
import os
import json
import faiss
import numpy as np
import openai
from sentence_transformers import SentenceTransformer
from textblob import TextBlob
from dotenv import load_dotenv
import re

from utils.embedding import load_model, load_dataset, get_question_embeddings, build_faiss_index
from utils.course_query import extract_course_info, get_course_by_code

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# UI setup
st.set_page_config(page_title="CrescentBot 🎓", layout="wide")
st.markdown("<h2 style='text-align: center;'>🎓 Crescent University Assistant</h2>", unsafe_allow_html=True)

# Style
st.markdown("""
    <style>
    .stChatMessage { padding: 10px 15px; border-radius: 15px; margin-bottom: 8px; max-width: 85%; }
    .user { background-color: #e0f7fa; margin-left: auto; }
    .bot { background-color: #f1f8e9; margin-right: auto; }
    </style>
""", unsafe_allow_html=True)


# --- Utility functions ---
@st.cache_resource
def initialize():
    model = load_model()
    qa_data, questions = load_dataset("data/crescent_qa.json")
    embeddings = get_question_embeddings(questions, model)
    index = build_faiss_index(embeddings)

    try:
        with open("data/course_data.json", "r", encoding="utf-8") as f:
            course_data = json.load(f)
    except FileNotFoundError:
        st.warning("⚠️ 'course_data.json' not found. Some course-specific queries may not work.")
        course_data = []

    return model, index, qa_data, questions, course_data


def correct_spelling(text):
    return str(TextBlob(text).correct())


def search(query, index, model, questions, qa_data, top_k=1):
    query_embedding = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    D, I = index.search(query_embedding, top_k)
    top_match_index = I[0][0]
    return qa_data[top_match_index]["answer"], float(D[0][0])


def gpt_fallback(prompt):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are CrescentBot, a helpful assistant for Crescent University students."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
            temperature=0.5,
        )
        return response.choices[0].message["content"].strip()
    except Exception as e:
        return "⚠️ Sorry, I'm currently unable to fetch a response from GPT-4."


# --- Initialize ---
model, index, qa_data, questions, course_data = initialize()


# --- Chat interface ---
if "history" not in st.session_state:
    st.session_state.history = []

user_input = st.chat_input("Ask me anything about Crescent University...")

if user_input:
    # Clear input
    st.session_state.history.append({"role": "user", "content": user_input})

    query = user_input.strip()
    course_info = extract_course_info(query, course_data)
    course_code_match = get_course_by_code(query, course_data)

    # Priority 1: Course-specific queries
    if course_info:
        response = course_info
    elif course_code_match:
        response = course_code_match
    else:
        # Try semantic search
        answer, score = search(query, index, model, questions, qa_data)
        if score > 0.6:
            response = answer
        else:
            response = gpt_fallback(query)

    st.session_state.history.append({"role": "bot", "content": response})


# --- Display chat history with bubbles ---
for msg in st.session_state.history:
    role_class = "user" if msg["role"] == "user" else "bot"
    st.markdown(f"<div class='stChatMessage {role_class}'>{msg['content']}</div>", unsafe_allow_html=True)
