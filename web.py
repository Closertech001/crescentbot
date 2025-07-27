# app.py - Crescent University Chatbot

import streamlit as st
import os
import json
import faiss
import numpy as np
import openai
from sentence_transformers import SentenceTransformer
from textblob import TextBlob
from symspellpy import SymSpell
from dotenv import load_dotenv
import re
import time

# Local modules
from utils.embedding import load_dataset, compute_question_embeddings
from utils.course_query import extract_course_query
from utils.semantic_search import load_chunks, build_index, search
from utils.greeting import is_greeting, greeting_responses, is_social_trigger, social_response
from utils.memory import init_memory
from log_utils import log_query

# --- Load environment variables ---
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# --- Load SymSpell for spell correction ---
sym_spell = SymSpell(max_dictionary_edit_distance=2)
sym_spell.load_dictionary("frequency_dictionary_en_82_765.txt", 0, 1)

# --- Load dataset and embeddings ---
dataset = load_dataset("data/crescent.json")
course_data = load_dataset("data/course_data.json")
question_embeddings = compute_question_embeddings(dataset)
index = build_index(question_embeddings)

# --- Initialize Streamlit app ---
st.set_page_config(page_title="Crescent University Chatbot", page_icon="🎓")
st.title("🎓 Crescent University Assistant")
st.caption("Your smart guide to everything Crescent University.")

# --- Session state initialization ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "related_questions" not in st.session_state:
    st.session_state.related_questions = []

if "last_department" not in st.session_state:
    st.session_state.last_department = None

init_memory()  # ✅ Initialize memory session state

# --- Chat input ---
user_input = st.chat_input("Ask me anything about Crescent University...")

if user_input:
    # Save chat
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    # ✅ Greeting check
    if is_greeting(user_input):
        response = greeting_responses()
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        st.rerun()

    # ✅ Social / Small Talk check
    if is_social_trigger(user_input):
        response = social_response(user_input)
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        st.rerun()

    # ✅ Extract course-level info (optional for academic query memory)
    query_info = extract_course_query(user_input)
    if query_info:
        st.session_state["last_query_info"] = query_info

    # ✅ Spell correction
    corrected_input = user_input
    suggestions = sym_spell.lookup_compound(user_input, max_edit_distance=2)
    if suggestions:
        corrected_input = suggestions[0].term

    # ✅ Sentiment
    sentiment = TextBlob(corrected_input).sentiment.polarity

    # ✅ Semantic Search
    response, score, matched_q = search(corrected_input, dataset, index)

    # ✅ Threshold for fallback
    threshold = 0.7

    if score < threshold:
        try:
            openai_response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant for Crescent University."},
                    {"role": "user", "content": corrected_input}
                ],
                temperature=0.5,
                max_tokens=300,
            )
            response = openai_response["choices"][0]["message"]["content"]
        except Exception as e:
            response = "Sorry, I'm currently unable to fetch a response from GPT-4."

    # ✅ Update memory
    st.session_state["last_query_info"] = {
        "query": corrected_input,
        "response": response,
        "score": score,
    }

    # ✅ Save assistant reply
    st.session_state.chat_history.append({"role": "assistant", "content": response})

    # ✅ Log query
    log_query(user_input, response, matched_q, score)

# --- Display chat history ---
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
