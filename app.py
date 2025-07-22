import streamlit as st
import os
import json
import faiss
import numpy as np
import openai
import sqlite3
from datetime import datetime
from sentence_transformers import SentenceTransformer
from textblob import TextBlob
from symspellpy import SymSpell
from dotenv import load_dotenv
import time

from utils.greetings import is_greeting
from utils.embedding import load_dataset, compute_question_embeddings
from utils.preprocess import normalize_input
from utils.search import semantic_search_faiss

# --- Load environment variables ---
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# --- Session state memory ---
if "history" not in st.session_state:
    st.session_state.history = []
if "user_dept" not in st.session_state:
    st.session_state.user_dept = None

# --- Load dataset and embeddings ---
qa_data, questions = load_dataset("data/qa_dataset.json")
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = compute_question_embeddings(model, questions)

# --- Build FAISS index ---
index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)

# --- Load SymSpell for typo correction ---
sym_spell = SymSpell(max_dictionary_edit_distance=2)
sym_spell.load_dictionary("data/frequency_dictionary_en_82_765.txt", 0, 1)

# --- SQLite logging ---
def log_to_sqlite(user_input, bot_response, matched_question, score, tone, dept):
    conn = sqlite3.connect("data/query_logs.db")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        user_input TEXT,
        bot_response TEXT,
        matched_question TEXT,
        score REAL,
        tone TEXT,
        department TEXT
    )''')
    c.execute("INSERT INTO logs (timestamp, user_input, bot_response, matched_question, score, tone, department) VALUES (?, ?, ?, ?, ?, ?, ?)",
              (datetime.now().isoformat(), user_input, bot_response, matched_question, score, tone, dept))
    conn.commit()
    conn.close()

# --- Detect tone ---
def detect_tone(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0.2:
        return "positive"
    elif polarity < -0.2:
        return "negative"
    else:
        return "neutral"

# --- OpenAI fallback ---
def gpt_fallback(query):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "system", "content": "You are a helpful assistant for Crescent University."},
                      {"role": "user", "content": query}],
            temperature=0.7,
            max_tokens=300
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return "⚠️ Sorry, I'm currently unable to fetch a response from GPT."

# --- Typo correction ---
def correct_typos(text):
    suggestions = sym_spell.lookup_compound(text, max_edit_distance=2)
    return suggestions[0].term if suggestions else text

# --- UI ---
st.set_page_config(page_title="CrescentBot 🎓", layout="centered")
st.title("🤖 CrescentBot – Your University Assistant")

st.markdown("""
Welcome! Ask me anything about Crescent University.  
Example:  
- _What are the courses in 200 level Accounting?_  
- _Who is the HOD of Computer Science?_  
""")

# --- Chat interface ---
user_input = st.chat_input("Ask me something about Crescent University...")

if user_input:
    st.chat_message("user").markdown(f"🧑‍🎓 **You:** {user_input}")

    # Greeting handler
    if is_greeting(user_input):
        bot_response = "👋 Hello! How can I help you today?"
        st.chat_message("assistant").markdown(bot_response)
        st.session_state.history.append(("You", user_input))
        st.session_state.history.append(("CrescentBot", bot_response))
        log_to_sqlite(user_input, bot_response, "", 1.0, "neutral", "")
    else:
        norm_query = normalize_input(correct_typos(user_input))
        matches = semantic_search_faiss(norm_query, model, index, qa_data, questions, top_k=3)

        top_match = matches[0] if matches else None
        confidence = top_match["score"] if top_match else 0

        if confidence > 0.7:
            bot_response = f"📘 *Here’s what I found for:* `{top_match['question']}`\n\n{top_match['answer']}"
            dept = top_match.get("department", "")
        else:
            fallback = gpt_fallback(user_input)
            bot_response = f"🤔 I wasn't sure, but here’s what I found:\n\n{fallback}"
            dept = ""

        st.chat_message("assistant").markdown(bot_response)

        tone = detect_tone(user_input)
        st.session_state.history.append(("You", user_input))
        st.session_state.history.append(("CrescentBot", bot_response))
        log_to_sqlite(user_input, bot_response, top_match["question"] if top_match else "", confidence, tone, dept)

# --- Chat history display ---
if st.session_state.history:
    with st.expander("📜 Conversation History", expanded=False):
        for role, msg in st.session_state.history:
            if role == "You":
                st.markdown(f"**🧑 {role}:** {msg}")
            else:
                st.markdown(f"**🤖 {role}:** {msg}")
