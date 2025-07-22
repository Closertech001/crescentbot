# app.py - CrescentBot Full App with Inline Utilities

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
import time
from datetime import datetime

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# --- Greeting logic ---
def get_greeting():
    hour = datetime.now().hour
    if 5 <= hour < 12:
        return "🌞 Good morning!"
    elif 12 <= hour < 17:
        return "🌤 Good afternoon!"
    elif 17 <= hour < 21:
        return "🌆 Good evening!"
    else:
        return "🌙 Hello!"

# --- Memory logic ---
def update_memory(memory, user_input, bot_response):
    memory.append({"user": user_input, "bot": bot_response})
    if len(memory) > 10:
        memory.pop(0)
    return memory

def get_last_context(memory):
    if memory:
        return memory[-1]["user"]
    return ""

# --- Spell correction ---
def correct_spelling(text):
    blob = TextBlob(text)
    return str(blob.correct())

# --- Dataset loading ---
def load_dataset(filepath="data/crescent_qa.json"):
    if not os.path.exists(filepath):
        st.error(f"❌ Dataset not found at `{filepath}`.")
        st.stop()
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    questions = [item["question"] for item in data]
    return data, questions

def load_course_data(filepath="data/course_data.json"):
    if not os.path.exists(filepath):
        st.error(f"❌ Course dataset not found at `{filepath}`.")
        st.stop()
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

# --- Semantic search ---
def semantic_search_faiss(query, model, index, questions, top_k=1):
    query_embedding = model.encode([query])[0]
    D, I = index.search(np.array([query_embedding]), top_k)
    results = [(questions[i], D[0][j]) for j, i in enumerate(I[0])]
    return results

# --- GPT fallback ---
def fallback_gpt_response(user_query, context=None):
    try:
        messages = [
            {"role": "system", "content": "You are CrescentBot, a helpful university assistant."}
        ]
        if context:
            messages.append({"role": "user", "content": f"Previously, the user said: {context}"})
        messages.append({"role": "user", "content": user_query})

        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages,
            temperature=0.5
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return "Sorry, I'm currently unable to fetch a response from GPT."

# --- Initialize model and embeddings ---
@st.cache_resource(show_spinner="🔍 Initializing model and embeddings...")
def initialize():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    qa_data, questions = load_dataset("data/crescent_qa.json")
    question_embeddings = model.encode(questions, convert_to_tensor=True)
    index = faiss.IndexFlatL2(question_embeddings.shape[1])
    index.add(np.array(question_embeddings))
    return model, index, qa_data, questions

# --- UI ---
st.set_page_config(page_title="CrescentBot 🎓", page_icon="🤖", layout="centered")

st.markdown("## 🤖 CrescentBot - Your Campus Assistant")
st.markdown("Ask me anything about Crescent University!")
st.markdown("---")

# --- Session state ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "memory" not in st.session_state:
    st.session_state.memory = []

# --- Load everything ---
model, index, qa_data, questions = initialize()
course_data = load_course_data("data/course_data.json")  # Optional if needed later

# --- User Input ---
user_query = st.text_input("You:", placeholder="What are the admission requirements?", key="input")

if user_query:
    with st.spinner("Thinking..."):
        greeting_response = get_greeting() if any(g in user_query.lower() for g in ["hello", "hi", "hey", "good morning", "good afternoon", "good evening"]) else None

        corrected_query = correct_spelling(user_query)
        results = semantic_search_faiss(corrected_query, model, index, questions, top_k=1)

        top_match, score = results[0]
        matched_entry = next((item for item in qa_data if item["question"] == top_match), None)

        if matched_entry and score < 1.0:
            bot_response = matched_entry["answer"]
        else:
            context = get_last_context(st.session_state.memory)
            bot_response = fallback_gpt_response(user_query, context)

        if greeting_response:
            bot_response = f"{greeting_response} {bot_response}"

        st.session_state.memory = update_memory(st.session_state.memory, user_query, bot_response)
        st.session_state.chat_history.append(("You", user_query))
        st.session_state.chat_history.append(("CrescentBot", bot_response))

# --- Display Chat ---
for sender, message in st.session_state.chat_history:
    with st.chat_message("user" if sender == "You" else "assistant"):
        st.markdown(message)
