# app.py

import streamlit as st
import openai
import numpy as np
from dotenv import load_dotenv
import os

from utils.embedding import load_model, load_qa_data, get_question_embeddings, build_faiss_index
from utils.semantic_search import search_faiss_index
from utils.greetings import (
    is_greeting,
    detect_sentiment,
    greeting_responses,
    is_small_talk,
    small_talk_response,
    extract_course_code,
    get_course_by_code
)
from utils.memory import update_memory, get_last_context

# --- Load environment variables ---
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# --- Page config ---
st.set_page_config(page_title="Crescent University Chatbot", page_icon="🎓")

# --- Session states ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "memory" not in st.session_state:
    st.session_state.memory = {}

# --- Load model and dataset ---
with st.spinner("🔄 Loading model and dataset..."):
    model = load_model()
    qa_data = load_qa_data()
    questions = [item["question"] for item in qa_data]
    embeddings = get_question_embeddings(questions, model)
    index = build_faiss_index(embeddings)

# --- Title ---
st.title("🎓 Crescent University Chatbot")

# --- Chat UI ---
user_query = st.chat_input("Ask me anything about Crescent University...")

if user_query:
    st.session_state.chat_history.append(("user", user_query))

    # --- Memory update ---
    st.session_state.memory = update_memory(st.session_state.memory, "last_query", user_query)

    # --- Greeting handling ---
    if is_greeting(user_query):
        response = greeting_responses(user_query)

    # --- Small talk ---
    elif is_small_talk(user_query):
        response = small_talk_response(user_query)

    # --- Course code match ---
    else:
        course_code = extract_course_code(user_query)
        if course_code:
            course_info = get_course_by_code(course_code, qa_data)
            if course_info:
                response = f"📘 *Here’s the info for* `{course_code}`:\n\n{course_info}"
            else:
                response = f"⚠️ I couldn't find any course with the code `{course_code}`."
        else:
            # --- FAISS Semantic Search ---
            match, score = search_faiss_index(user_query, model, index, qa_data, questions)
            if match and score > 0.6:
                response = f"🔍 *Here’s what I found:*\n\n{match}"
            else:
                # --- OpenAI fallback ---
                try:
                    completion = openai.ChatCompletion.create(
                        model="gpt-4",
                        messages=[{"role": "system", "content": "You are a helpful assistant for Crescent University."},
                                  {"role": "user", "content": user_query}]
                    )
                    response = completion.choices[0].message.content
                except Exception:
                    response = "❌ Sorry, I couldn't retrieve a response at the moment."

    st.session_state.chat_history.append(("bot", response))

# --- Display chat history ---
for sender, msg in st.session_state.chat_history:
    if sender == "user":
        st.markdown(f"👤 **You:** {msg}", unsafe_allow_html=True)
    else:
        st.markdown(f"🤖 **CrescentBot:** {msg}", unsafe_allow_html=True)

# --- Optional persistent memory (SQLite, not implemented here) ---
# Future toggle: Add support for `db.save_memory(user_id, memory)` and reload across sessions
