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

# --- Load environment variables ---
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# --- Load dataset ---
@st.cache_resource
def load_dataset():
    with open("data/crescent_qa.json", "r", encoding="utf-8") as f:
        return json.load(f)

@st.cache_resource
def load_course_data():
    with open("data/course_data.json", "r", encoding="utf-8") as f:
        return json.load(f)

# --- Compute embeddings ---
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def compute_question_embeddings(data, model):
    questions = [item["question"] for item in data]
    embeddings = model.encode(questions, show_progress_bar=True)
    return np.array(embeddings)

# --- FAISS Index ---
@st.cache_resource
def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

# --- Spell Correction ---
@st.cache_resource
def load_symspell():
    sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
    dict_path = "frequency_dictionary_en_82_765.txt"
    sym_spell.load_dictionary(dict_path, term_index=0, count_index=1)
    return sym_spell

def correct_query(sym_spell, query):
    suggestion = sym_spell.lookup_compound(query, max_edit_distance=2)
    return suggestion[0].term if suggestion else query

# --- Tone Detection ---
def detect_tone(query):
    blob = TextBlob(query)
    polarity = blob.sentiment.polarity
    if polarity > 0.3:
        return "positive"
    elif polarity < -0.3:
        return "negative"
    else:
        return "neutral"

# --- Emotion Detection ---
def detect_emotion(query):
    emotions = {
        "happy": ["happy", "glad", "great", "awesome", "fantastic", "😊"],
        "sad": ["sad", "down", "depressed", "unhappy", "😢"],
        "angry": ["angry", "mad", "frustrated", "annoyed", "😠"],
        "confused": ["confused", "lost", "don't get it", "stuck", "🤔"],
    }
    for emotion, keywords in emotions.items():
        if any(kw in query.lower() for kw in keywords):
            return emotion
    return None

# --- Greeting Detection ---
def detect_greeting(query):
    greetings = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"]
    return any(greet in query.lower() for greet in greetings)

# --- Name Detection ---
def detect_name(query):
    name_patterns = [
        r"i'?m\s+([A-Z][a-z]+)",
        r"my name is\s+([A-Z][a-z]+)",
        r"it's\s+([A-Z][a-z]+)",
    ]
    for pattern in name_patterns:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            return match.group(1)
    return None

# --- Semantic Search ---
def search_semantic(query, index, model, data, top_k=1):
    query_embedding = model.encode([query])
    D, I = index.search(query_embedding, top_k)
    if I[0][0] < len(data):
        return data[I[0][0]], 1 - D[0][0]
    return None, 0.0

# --- GPT fallback ---
def gpt_fallback(query):
    try:
        res = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful university assistant."},
                {"role": "user", "content": query},
            ],
            temperature=0.4,
        )
        return res["choices"][0]["message"]["content"]
    except Exception:
        res = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful university assistant."},
                {"role": "user", "content": query},
            ],
            temperature=0.4,
        )
        return res["choices"][0]["message"]["content"]

# --- Streamlit UI ---
st.set_page_config(page_title="CrescentBot 🤖", page_icon="🌙")
st.markdown("""
    <style>
    .message-user { background-color: #DCF8C6; padding: 8px; border-radius: 10px; margin-bottom: 5px; }
    .message-bot { background-color: #F1F0F0; padding: 8px; border-radius: 10px; margin-bottom: 5px; }
    </style>
""", unsafe_allow_html=True)

st.title("🌙 Crescent University Chatbot")
model = load_model()
data = load_dataset()
courses = load_course_data()
embeddings = compute_question_embeddings(data, model)
index = build_faiss_index(embeddings)
sym_spell = load_symspell()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "user_name" not in st.session_state:
    st.session_state.user_name = None

# --- Chat Loop ---
for msg in st.session_state.messages:
    role = msg["role"]
    content = msg["content"]
    css_class = "message-user" if role == "user" else "message-bot"
    st.markdown(f'<div class="{css_class}">{content}</div>', unsafe_allow_html=True)

query = st.chat_input("Ask me anything about Crescent University... 🏫")
if query:
    st.session_state.messages.append({"role": "user", "content": query})
    query = correct_query(sym_spell, query)

    # Greeting
    if detect_greeting(query):
        greeting = "👋 Hello again! How can I help you today?"
        if st.session_state.user_name:
            greeting = f"👋 Hello again, {st.session_state.user_name}! How can I help you today?"
        st.session_state.messages.append({"role": "assistant", "content": greeting})

    # Name
    elif name := detect_name(query):
        st.session_state.user_name = name
        st.session_state.messages.append({"role": "assistant", "content": f"😊 Nice to meet you, {name}! How can I assist you today?"})

    # Gratitude
    elif "thank" in query.lower():
        st.session_state.messages.append({"role": "assistant", "content": "🙏 You're most welcome!"})

    # Farewell
    elif any(f in query.lower() for f in ["bye", "see you", "goodbye"]):
        st.session_state.messages.append({"role": "assistant", "content": "👋 Bye for now! Feel free to ask anything anytime."})

    # Emotion
    elif emotion := detect_emotion(query):
        response_map = {
            "happy": "😄 I'm glad you're feeling good! Let me know how I can help further.",
            "sad": "😢 I'm here for you. Let me know how I can assist or make things better.",
            "angry": "😠 I understand your frustration. Let me help solve the issue.",
            "confused": "🤔 Let's work through this together. Could you tell me a bit more?",
        }
        st.session_state.messages.append({"role": "assistant", "content": response_map[emotion]})

    else:
        top_match, score = search_semantic(query, index, model, data, top_k=1)
        if top_match and score > 0.6:
            st.session_state.messages.append({"role": "assistant", "content": top_match["answer"]})
        else:
            response = gpt_fallback(query)
            st.session_state.messages.append({"role": "assistant", "content": response})
