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

# --- Internal utility imports ---
from utils.greetings import is_greeting, get_greeting_response
from utils.preprocess import normalize_input
from utils.memory import store_context_from_query, enrich_query_with_context
from utils.tone import detect_tone, get_tone_response
from utils.semantic_search import load_chunks, build_index, search
from log_utils import log_query  # for logging user interactions

# --- Load environment variables ---
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# --- Load resources ---
@st.cache_resource
def get_model_and_index():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    chunks = load_chunks()
    index = build_index(model, chunks)
    return model, chunks, index

model, chunks, index = get_model_and_index()

# --- Streamlit UI config ---
st.set_page_config(page_title="Crescent University Chatbot", page_icon="🎓", layout="centered")
st.title("🎓 Crescent University Chatbot")
st.markdown("Ask me anything about Crescent University!")

# --- Initialize memory ---
if "history" not in st.session_state:
    st.session_state["history"] = []

# --- User input ---
user_input = st.chat_input("Type your question here...")

if user_input:
    st.chat_message("user").write(user_input)

    # Handle greetings
    if is_greeting(user_input):
        response = get_greeting_response()
        st.chat_message("assistant").write(response)
        store_context_from_query(user_input, st.session_state, response)
        log_query(user_input, response)
        st.stop()

    # Normalize and enrich input
    norm_query = normalize_input(user_input)
    enriched_query = enrich_query_with_context(norm_query, st.session_state)

    # --- Semantic Search ---
    top_match, score = search(enriched_query, index, model, chunks, top_k=1)

    # --- Fallback to GPT if low confidence ---
    if score < 0.6:
        try:
            completion = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "user", "content": enriched_query}],
                max_tokens=500,
                temperature=0.7
            )
            response = completion.choices[0].message["content"].strip()
        except Exception as e:
            response = "Sorry, I'm currently unable to fetch a response from GPT-4."
    else:
        response = top_match

    # --- Detect tone and adjust if needed ---
    tone = detect_tone(user_input)
    response = get_tone_response(response, tone)

    # --- Display and store ---
    st.chat_message("assistant").write(response)
    store_context_from_query(user_input, st.session_state, response)
    log_query(user_input, response)
