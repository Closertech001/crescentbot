# app.py

import streamlit as st
import os
import json
import faiss
import numpy as np
import openai
from sentence_transformers import SentenceTransformer
from textblob import TextBlob
from dotenv import load_dotenv
import time

from utils.embedding import load_dataset, compute_question_embeddings
from utils.course_query import extract_course_query
from utils.semantic_search import semantic_search

# --- App Settings ---
st.set_page_config(page_title="CrescentBot 🎓", layout="centered")

# --- Custom CSS ---
with open("assets/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# --- Load Environment ---
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# --- Initialize session state for chat history ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Helper to detect tone ---
def detect_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0.2:
        return "positive"
    elif polarity < -0.2:
        return "negative"
    else:
        return "neutral"

# --- Display message as chat bubble ---
def display_chat(message, role="bot"):
    if role == "user":
        st.markdown(f"""
        <div class="user-bubble">
            <strong>👤 You:</strong><br>{message}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="bot-bubble">
            <strong>🤖 CrescentBot:</strong><br>{message}
        </div>
        """, unsafe_allow_html=True)

# --- GPT Fallback ---
def fallback_response(prompt):
    try:
        res = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.7,
        )
        return res.choices[0].message.content.strip()
    except Exception:
        return "Sorry, I'm currently unable to fetch a response from GPT."

# --- Load all resources ---
@st.cache_resource(show_spinner="🔍 Loading knowledge base...")
def initialize():
    try:
        qa_data = load_dataset("data/crescent_qa.json")
        questions = [item["question"] for item in qa_data]
        embeddings = compute_question_embeddings(questions)
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)

        with open("data/course_data.json") as f:
            course_data = json.load(f)

        model = SentenceTransformer("all-MiniLM-L6-v2")
        return model, index, qa_data, questions, course_data

    except FileNotFoundError as e:
        st.error(f"❌ {str(e).split(':')[-1].strip()} not found. Please upload the file to the app directory.")
        return None, None, [], [], {}

# --- Header ---
st.markdown("<h2 style='text-align:center;'>🎓 CrescentBot — Your University Assistant</h2>", unsafe_allow_html=True)

# --- Input ---
query = st.chat_input("Ask anything about Crescent University...")

if query:
    # Clear chat input
    st.session_state.chat_history.append({"role": "user", "content": query})
    sentiment = detect_sentiment(query)

    model, index, qa_data, questions, course_data = initialize()

    if model is None or index is None or not qa_data:
        st.error("❌ Cannot respond: Knowledge base not loaded.")
        st.stop()

    with st.spinner("🤖 Thinking..."):
        # Check if it's a course query
        course_code, course_info = extract_course_query(query, course_data)
        if course_info:
            response = f"📘 *Here’s the info for* `{course_code}`:\n\n{course_info}"
        else:
            answer, matched_question, score = semantic_search(query, index, model, questions, qa_data)
            if score > 50:
                response = fallback_response(query)
            else:
                response = f"**Matched:** `{matched_question}`\n\n📘 {answer}"

        st.session_state.chat_history.append({"role": "bot", "content": response})

# --- Display chat history ---
for msg in st.session_state.chat_history:
    display_chat(msg["content"], msg["role"])
