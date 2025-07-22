import streamlit as st
import openai
import os
import json
import faiss
import numpy as np
from datetime import datetime
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# --- ENV SETUP ---
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# --- UTILITY FUNCTIONS ---

# Load SentenceTransformer model
@st.cache_resource
def load_model(model_name="all-MiniLM-L6-v2"):
    return SentenceTransformer(model_name)

# Load QA data from JSON
def load_dataset(filepath="data/qa_dataset.json"):
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    questions = [item["question"] for item in data]
    return data, questions

# Compute embeddings
def get_question_embeddings(questions, model):
    return model.encode(questions, convert_to_numpy=True, normalize_embeddings=True)

# Build FAISS index
def build_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    return index

# Search for most relevant answers
def semantic_search_faiss(query, model, index, qa_data, questions, top_k=3):
    query_embedding = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    scores, indices = index.search(query_embedding, top_k)
    results = []
    for i, idx in enumerate(indices[0]):
        if idx < len(qa_data):
            result = qa_data[idx].copy()
            result["score"] = float(scores[0][i])
            result["matched_question"] = questions[idx]
            results.append(result)
    return results

# Dynamic greeting based on time
def get_greeting():
    hour = datetime.now().hour
    if 5 <= hour < 12:
        return "Good morning"
    elif 12 <= hour < 17:
        return "Good afternoon"
    elif 17 <= hour < 21:
        return "Good evening"
    else:
        return "Hello"

# Session memory (chat history)
def update_memory(chat_history, user_msg, bot_msg):
    chat_history.append({"user": user_msg, "bot": bot_msg})

def get_last_context(chat_history):
    return chat_history[-1] if chat_history else None

# --- INITIALIZE ---
@st.cache_resource
def initialize():
    model = load_model()
    qa_data, questions = load_dataset()
    embeddings = get_question_embeddings(questions, model)
    index = build_faiss_index(embeddings)
    return model, index, qa_data, questions

model, index, qa_data, questions = initialize()

# --- PAGE CONFIG ---
st.set_page_config(page_title="CrescentBot 🎓", page_icon="🤖", layout="centered")
st.markdown("<h1 style='text-align: center;'>🤖 CrescentBot</h1>", unsafe_allow_html=True)
st.markdown("Ask me anything about Crescent University!")

# --- SESSION STATE ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- GREETING ---
if len(st.session_state.chat_history) == 0:
    greeting = get_greeting()
    st.chat_message("assistant").markdown(f"{greeting} 👋\n\nI'm CrescentBot! Ask me anything about the university.")

# --- INPUT FIELD ---
user_input = st.chat_input("Ask something about Crescent University...")

# --- PROCESS INPUT ---
if user_input:
    st.chat_message("user").markdown(user_input)

    # Retrieve last memory (optional for context tracking)
    last_context = get_last_context(st.session_state.chat_history)

    # Semantic Search
    matches = semantic_search_faiss(user_input, model, index, qa_data, questions, top_k=3)
    top = matches[0] if matches else None

    if top and top["score"] > 0.5:
        response = f"📘 **Answer:** {top['answer']}"
        if top.get("department"):
            response += f"\n\n🏛️ *Department:* {top['department']}"
    else:
        response = "🤔 Sorry, I couldn't find an exact answer. Could you try rephrasing?"

    st.chat_message("assistant").markdown(response)
    update_memory(st.session_state.chat_history, user_input, response)
