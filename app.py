# app.py

import streamlit as st
import os
import json
import faiss
import openai
import numpy as np
from dotenv import load_dotenv

from utils.embedding import load_model, load_dataset, get_question_embeddings, build_faiss_index
from utils.search import semantic_search_faiss
from utils.greetings import get_greeting
from utils.memory import update_memory, get_last_context

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Page config
st.set_page_config(page_title="🎓 CrescentBot - University Assistant", page_icon="🤖")
st.title("🤖 CrescentBot - Your Crescent University Assistant")
st.markdown("Ask me anything about Crescent University! 📘")

# Session state init
if "memory" not in st.session_state:
    st.session_state.memory = {}

# Load data, model, and build index
@st.cache_resource
def init_system():
    model = load_model()
    qa_data, questions = load_dataset("data/crescent_qa.json")
    question_embeddings = get_question_embeddings(questions, model)
    index = build_faiss_index(question_embeddings)
    return model, qa_data, questions, index

model, qa_data, questions, index = init_system()

# Sidebar
with st.sidebar:
    st.markdown("### 👋 Greeting")
    st.markdown(get_greeting())
    st.markdown("### 🧠 Persistent Memory")
    st.json(st.session_state.memory)

# Chat input
user_query = st.chat_input("Type your question here...")

if user_query:
    st.chat_message("user").markdown(f"💬 {user_query}")

    # Store last question in memory
    update_memory(st.session_state.memory, "last_query", user_query)

    # Semantic search
    results = semantic_search_faiss(user_query, model, index, qa_data, questions, top_k=3)

    # Check confidence threshold
    top_match = results[0] if results else None
    if top_match and top_match["score"] > 0.6:
        answer = top_match["answer"]
        dept = top_match.get("department", "")
        response = f"📘 *Here’s what I found*:\n\n**Answer:** {answer}\n\n_Department: {dept}_"
    else:
        # Fallback to GPT
        try:
            gpt_response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant for Crescent University. Be brief and clear."},
                    {"role": "user", "content": user_query}
                ],
                max_tokens=150,
                temperature=0.4
            )
            answer = gpt_response['choices'][0]['message']['content'].strip()
            response = f"🤖 *Here's a GPT-suggested answer:*\n\n{answer}"
        except Exception as e:
            response = "⚠️ Sorry, I couldn’t fetch a response right now."

    update_memory(st.session_state.memory, "last_response", response)
    st.chat_message("assistant").markdown(response)
