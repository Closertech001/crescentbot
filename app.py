# app.py - Crescent University Chatbot

import streamlit as st
import os
import json
import openai
import faiss
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from textblob import TextBlob

from utils.embedding import load_dataset, compute_question_embeddings
from utils.semantic_search import load_chunks, build_index, search
from utils.symspell_corrector import correct_text
from utils.course_query import extract_course_query
from utils.openai_fallback import ask_openai
from utils.greetings import detect_greeting, get_greeting_response
from utils.memory import update_memory, get_last_context
from utils.preprocess import normalize_input  # NEW

# --- Load environment variables ---
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# --- Load data and model ---
chunks = load_chunks("data/qa_dataset.json")
model = SentenceTransformer("all-MiniLM-L6-v2")
index = build_index(chunks, model)

# --- UI Setup ---
st.set_page_config(page_title="CrescentBot 🤖", layout="wide")
st.title("🎓 Crescent University Chat Assistant")
st.markdown("Ask me anything about Crescent University. I'm here to help! 🤝")

# --- In-session memory ---
if "memory" not in st.session_state:
    st.session_state.memory = {}

# --- Chat interface ---
query = st.text_input("You:", placeholder="e.g., What are the courses in 200 level law?", key="input")

if query:
    # Normalize + correct
    normalized = normalize_input(query)  # 👈 Use normalization
    corrected = correct_text(normalized)
    st.markdown(f"🛠️ Corrected: `{corrected}`")

    # Greeting handling
    if detect_greeting(corrected):
        st.markdown(get_greeting_response())
    else:
        # Search
        top_match, score = search(corrected, index, model, chunks, top_k=1)

        # Use OpenAI fallback if confidence is low
        if score < 0.65:
            response = ask_openai(corrected)
        else:
            # Course-specific format
            course_code = extract_course_query(corrected)
            if course_code:
                response = f"📘 *Here’s the info for* `{course_code}`:\n\n{top_match['answer']}"
            else:
                response = top_match['answer']

        st.markdown(f"**CrescentBot:** {response}")

        # Save last response in memory
        update_memory(st.session_state.memory, "last_response", response)

    # Optional: show debug info
    # st.json({
    #     "normalized": normalized,
    #     "corrected": corrected,
    #     "score": score,
    #     "top_question": top_match['question'],
    #     "memory": st.session_state.memory
    # })
