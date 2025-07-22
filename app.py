# app.py – CrescentBot with FAISS support

import streamlit as st
import json
import openai
import numpy as np
import faiss
from datetime import datetime
from sentence_transformers import SentenceTransformer

from utils.embedding import load_model, load_qa_data
from utils.course_query import extract_course_query
from utils.department_query import extract_department_query
from utils.memory import update_memory, get_last_context
from utils.style import set_custom_styles
from utils.fallback import ask_gpt_fallback
from utils.tone import detect_greeting, detect_small_talk

# --- Load secrets ---
openai.api_key = st.secrets["OPENAI_API_KEY"]

# --- Title ---
st.set_page_config(page_title="CrescentBot", page_icon="🎓")
set_custom_styles()
st.title("🎓 CrescentBot - Your University Assistant")

# --- Session State ---
if "memory" not in st.session_state:
    st.session_state.memory = {}

# --- Load data ---
@st.cache_resource
def load_all_resources():
    model = load_model("all-MiniLM-L6-v2")
    qa_data = load_qa_data()
    questions = [item["question"] for item in qa_data]
    embeddings = model.encode(questions, convert_to_numpy=True, normalize_embeddings=True)
    
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    
    return model, qa_data, questions, index, embeddings

model, qa_data, qa_questions, faiss_index, qa_embeddings_np = load_all_resources()

# --- Search ---
def semantic_search(query, top_k=3, min_score=0.6):
    query_embedding = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    scores, indices = faiss_index.search(query_embedding, top_k)
    score = scores[0][0]
    if score >= min_score:
        idx = indices[0][0]
        return qa_data[idx]["answer"]
    return None

# --- Chatbot UI ---
user_input = st.chat_input("Ask me anything about Crescent University...")

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)

    # --- Handle greetings or small talk ---
    if detect_greeting(user_input) or detect_small_talk(user_input):
        response = "👋 Hello! I’m CrescentBot, your university assistant. How can I help you today?"
        st.chat_message("assistant").markdown(response)

    else:
        # --- Department memory ---
        last_dept = get_last_context(st.session_state.memory, "last_department")

        # --- Extract info ---
        course_code = extract_course_query(user_input)
        department = extract_department_query(user_input)

        if department:
            st.session_state.memory = update_memory(st.session_state.memory, "last_department", department)

        # --- Course Code Handler ---
        if course_code:
            response = f"📘 *Here’s the info for* `{course_code}`:\n\nSorry, I don’t have details about this course yet."
        
        # --- Semantic Search (FAISS) ---
        else:
            semantic_answer = semantic_search(user_input)
            if semantic_answer:
                response = semantic_answer
            else:
                response = ask_gpt_fallback(user_input)

        # --- Display Response ---
        with st.chat_message("assistant"):
            st.markdown(response)

