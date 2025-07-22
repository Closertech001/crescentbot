import streamlit as st
import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from textblob import TextBlob
from symspellpy import SymSpell
from dotenv import load_dotenv
import re
import time

from utils.embedding import load_model, load_dataset, get_question_embeddings, build_faiss_index
from utils.course_query import extract_course_query, get_course_info

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Page config
st.set_page_config(page_title="CrescentBot 🎓", page_icon="🤖", layout="centered")

# Style injection
with open("assets/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Load SentenceTransformer model and data
@st.cache_resource
def setup_embeddings():
    model = load_model()
    qa_data, questions = load_dataset("data/crescent_qa.json")
    embeddings = get_question_embeddings(questions, model)
    index = build_faiss_index(embeddings)
    return model, qa_data, questions, embeddings, index

model, qa_data, questions, embeddings, index = setup_embeddings()

# Load course data
with open("data/course_data.json", "r", encoding="utf-8") as f:
    course_data = json.load(f)

# SymSpell for typo correction
@st.cache_resource
def setup_symspell():
    sym_spell = SymSpell(max_dictionary_edit_distance=2)
    dictionary_path = os.path.join(os.path.dirname(__file__), "utils/frequency_dictionary_en_82_765.txt")
    sym_spell.load_dictionary(dictionary_path, 0, 1)
    return sym_spell

sym_spell = setup_symspell()

def correct_typos(text):
    corrected = sym_spell.lookup_compound(text, max_edit_distance=2)
    return corrected[0].term if corrected else text

def detect_tone(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0.5:
        return "humorous"
    elif polarity > 0.1:
        return "casual"
    else:
        return "formal"

def style_response(text, tone="formal"):
    if tone == "humorous":
        return f"😄 *Hehe, here's a fun one for you:*\n\n{text}"
    elif tone == "casual":
        return f"🧢 *Sure thing!*\n\n{text}"
    else:
        return f"📘 {text}"

def semantic_search(query, index, model, questions, qa_data, top_k=1):
    query_embedding = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    distances, indices = index.search(query_embedding, top_k)
    top_idx = indices[0][0]
    return qa_data[top_idx], float(distances[0][0])

def generate_course_response(user_input):
    info = extract_course_query(user_input)
    courses = get_course_info("data/course_data.json", **info)
    if not courses:
        return "Sorry, I couldn't find any courses matching your query. Please check your level, semester, or department."

    grouped = {}
    for course in courses:
        dept = course["department"]
        grouped.setdefault(dept, []).append(f"- `{course['code']}`: {course['title']}")

    messages = []
    for dept, course_list in grouped.items():
        messages.append(f"📚 **{dept} Courses:**\n" + "\n".join(course_list))
    return "\n\n".join(messages)

def handle_query(user_query):
    normalized = correct_typos(user_query)
    tone = detect_tone(user_query)

    if re.search(r"\b(course|semester|level|department)\b", normalized, re.IGNORECASE):
        response = generate_course_response(normalized)
    else:
        match, score = semantic_search(normalized, index, model, questions, qa_data)
        response = match["answer"] if score > 0.55 else "Hmm, I'm not sure about that. Try rephrasing?"

    return style_response(response, tone)

# UI
st.markdown("<h2 style='text-align: center;'>🎓 Crescent University Chatbot</h2>", unsafe_allow_html=True)
st.markdown("<div class='chatbox'>", unsafe_allow_html=True)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Show history
for role, message in st.session_state.chat_history:
    if role == "user":
        st.markdown(f"<div class='user-bubble'>{message}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='bot-bubble'>{message}</div>", unsafe_allow_html=True)

# Input
with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input("Ask CrescentBot something...", placeholder="E.g., What courses are offered in 200 level second semester Law?", label_visibility="collapsed")
    submitted = st.form_submit_button("Send")

if submitted and user_input:
    st.session_state.chat_history.append(("user", user_input))
    with st.spinner("CrescentBot is typing..."):
        time.sleep(0.8)
        bot_response = handle_query(user_input)
    st.session_state.chat_history.append(("bot", bot_response))
    st.rerun()

st.markdown("</div>", unsafe_allow_html=True)
