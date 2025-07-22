import streamlit as st
import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from utils.embedding import load_model, load_dataset, get_question_embeddings, build_faiss_index
from utils.course_query import extract_course_query
from utils.semantic_search import semantic_search

load_dotenv()
st.set_page_config(page_title="CrescentBot 🤖", layout="wide")

# Apply custom CSS
with open("assets/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# --- Load Course Data ---
@st.cache_data
def load_course_data(filepath="data/course_data.json"):
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        st.warning("⚠️ 'course_data.json' not found. Some course-specific queries may not work.")
        return {}

# --- Initialization ---
@st.cache_resource
def initialize():
    qa_data, questions = load_dataset("data/crescent_qa.json")
    model = load_model()
    embeddings = get_question_embeddings(questions, model)
    index = build_faiss_index(embeddings)
    course_data = load_course_data()
    return model, index, qa_data, questions, course_data

model, index, qa_data, questions, course_data = initialize()

# --- App Title ---
st.title("🎓 CrescentBot - Ask Me Anything!")

# --- Initialize chat history ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Display chat history ---
for msg in st.session_state.messages:
    role = msg["role"]
    content = msg["content"]
    align = "user" if role == "user" else "bot"
    st.markdown(f"<div class='chat-bubble {align}'>{content}</div>", unsafe_allow_html=True)

# --- Chat input ---
user_input = st.chat_input("Type your question here...")

# --- Handle user query ---
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    # --- Check for course-specific query ---
    course_code, course_info = extract_course_query(user_input, course_data)
    if course_code and course_info:
        response = f"📘 *Here’s the info for* `{course_code}`:\n\n{course_info}"
    else:
        # --- Semantic search from dataset ---
        answer, matched_q, score = semantic_search(
            user_input, index, model, questions, qa_data, top_k=1
        )

        if score > 0.5:
            response = answer
        else:
            response = "🤔 I’m not sure about that yet. Could you rephrase your question?"

    # Add bot response to session
    st.session_state.messages.append({"role": "bot", "content": response})

    # Rerun to clear input box
    st.rerun()
