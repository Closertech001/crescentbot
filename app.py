import streamlit as st
import openai
import json
import numpy as np
import faiss
import os
import sqlite3
import time
import re
from sentence_transformers import SentenceTransformer
from utils.embedding import load_qa_data, get_question_embeddings, build_faiss_index
from utils.course_query import extract_course_info, get_course_by_code
from textblob import TextBlob
from dotenv import load_dotenv

# --- Load environment variables ---
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# --- Load model and index ---
@st.cache_resource
def load_resources():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    data = load_qa_data()
    questions = [item["question"] for item in data]
    answers = [item["answer"] for item in data]
    embeddings = get_question_embeddings(questions, model)
    index = build_faiss_index(np.array(embeddings))
    return model, data, index, answers

model, data, index, answers = load_resources()

# --- SQLite Memory ---
def get_memory():
    conn = sqlite3.connect("memory.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS memory (
            session_id TEXT PRIMARY KEY,
            department TEXT,
            level TEXT,
            semester TEXT
        )
    """)
    conn.commit()
    return conn

def update_memory(conn, session_id, department=None, level=None, semester=None):
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM memory WHERE session_id=?", (session_id,))
    row = cursor.fetchone()
    if row:
        cursor.execute("""
            UPDATE memory SET department=?, level=?, semester=? WHERE session_id=?
        """, (department, level, semester, session_id))
    else:
        cursor.execute("""
            INSERT INTO memory (session_id, department, level, semester)
            VALUES (?, ?, ?, ?)
        """, (session_id, department, level, semester))
    conn.commit()

def retrieve_memory(conn, session_id):
    cursor = conn.cursor()
    cursor.execute("SELECT department, level, semester FROM memory WHERE session_id=?", (session_id,))
    row = cursor.fetchone()
    return {"department": row[0], "level": row[1], "semester": row[2]} if row else {}

# --- FAISS Semantic Search ---
def search(query, index, model, data, top_k=1):
    query_embedding = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    D, I = index.search(query_embedding, top_k)
    return data[I[0][0]], D[0][0]  # Return top match and score

# --- Streamlit UI ---
st.set_page_config(page_title="CrescentBot 🎓", layout="centered")
st.title("🤖 Crescent University Assistant")

if "history" not in st.session_state:
    st.session_state.history = []

session_id = "user_session"
conn = get_memory()
memory = retrieve_memory(conn, session_id)

prompt = st.chat_input("Ask me anything about Crescent University...")

if prompt:
    st.session_state.history.append(("user", prompt))
    
    # Typo correction
    corrected = str(TextBlob(prompt).correct())
    if corrected.lower() != prompt.lower():
        prompt = corrected

    # Course code response
    course_code_result = get_course_by_code(prompt, json.load(open("data/course_data.json")))
    if course_code_result and "couldn't find" not in course_code_result:
        response = f"📘 {course_code_result}"
    else:
        # Check if it's a course info request
        course_response = extract_course_info(prompt, json.load(open("data/course_data.json")), memory)
        if isinstance(course_response, str):
            response = f"📚 {course_response}"
        else:
            # Semantic fallback
            match, score = search(prompt, index, model, answers)
            if score < 0.6:
                response = "🤔 Sorry, I couldn't find an exact answer, but I'm learning!"
            else:
                response = f"💡 {match['answer']}"

    # Save updated memory
    update_memory(conn, session_id, memory.get("department"), memory.get("level"), memory.get("semester"))

    st.session_state.history.append(("bot", response))

# --- Display chat ---
for role, msg in st.session_state.history:
    if role == "user":
        with st.chat_message("user"):
            st.markdown(f"**You:** {msg}")
    else:
        with st.chat_message("assistant"):
            st.markdown(msg)
