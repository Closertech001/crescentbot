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
from datetime import datetime
import re
import time

from utils.embedding import load_model, load_dataset, get_question_embeddings, build_faiss_index
from utils.course_query import extract_course_query, get_course_response
from utils.memory import MemoryManager
from utils.feedback_logger import log_feedback
from utils.clean_text import normalize_text, correct_text
from utils.greetings import get_smalltalk_response

# --- Load environment variables ---
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# --- Initialize memory manager ---
memory = MemoryManager()

# --- Load SymSpell for typo correction ---
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
sym_spell.load_dictionary("data/frequency_dictionary_en_82_765.txt", term_index=0, count_index=1)

# --- Helper functions ---
def detect_tone(text):
    sentiment = TextBlob(text).sentiment.polarity
    if sentiment > 0.2:
        return "positive"
    elif sentiment < -0.2:
        return "negative"
    return "neutral"

def typewriter_effect(text, delay=0.015):
    output = ""
    for char in text:
        output += char
        st.markdown(output)
        time.sleep(delay)

def generate_response(user_query, top_match, score, course_response):
    if course_response:
        return course_response
    elif score > 0.7:
        return top_match.get("answer", "I'm not sure about that.")
    else:
        # fallback to OpenAI GPT
        try:
            prompt = f"User: {user_query}\nAssistant:"
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "system", "content": "You are a helpful assistant for Crescent University."},
                          {"role": "user", "content": user_query}]
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return "Sorry, I'm currently unable to fetch a response from GPT-4."

@st.cache_resource(show_spinner="Loading models and index...")
def initialize():
    model = load_model()
    qa_data, questions = load_dataset("data/crescent_qa.json")
    embeddings = get_question_embeddings(questions, model)
    index = build_faiss_index(embeddings)

    try:
        with open("data/course_data.json", "r", encoding="utf-8") as f:
            course_data = json.load(f)
    except FileNotFoundError:
        st.warning("⚠️ 'course_data.json' not found. Some course-specific queries may not work.")
        course_data = []

    return model, index, qa_data, questions, course_data

# --- Main App ---
st.set_page_config(page_title="CrescentBot 🎓", page_icon="🤖")
st.markdown("""
    <style>
    .stChatMessage.user {background-color: #d1e7dd; color: #000; border-radius: 12px; padding: 10px; margin: 5px 0;}
    .stChatMessage.assistant {background-color: #f8d7da; color: #000; border-radius: 12px; padding: 10px; margin: 5px 0;}
    </style>
""", unsafe_allow_html=True)

st.title("🎓 Crescent University Chatbot")

model, index, qa_data, questions, course_data = initialize()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_query = st.chat_input("Ask me anything about Crescent University")
if user_query:
    st.session_state.chat_history.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    norm_query = normalize_text(user_query)
    corrected_query = correct_text(norm_query, sym_spell)

    course_code = extract_course_query(corrected_query)
    course_response = get_course_response(course_code, course_data) if course_code else None

    query_embedding = model.encode([corrected_query], convert_to_numpy=True, normalize_embeddings=True)
    top_k = 1
    D, I = index.search(query_embedding, top_k)
    top_match = qa_data[I[0][0]] if I[0][0] < len(qa_data) else None
    score = D[0][0]

    tone = detect_tone(user_query)

    with st.chat_message("assistant"):
        with st.spinner("CrescentBot is thinking..."):
            response = generate_response(user_query, top_match, score, course_response)

            if tone == "positive":
                response = "😊 " + response
            elif tone == "negative":
                response = "😟 " + response
            else:
                response = "🤖 " + response

            st.markdown(response)

    st.session_state.chat_history.append({"role": "assistant", "content": response})

    log_feedback(user_query, response, tone)
