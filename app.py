import streamlit as st
import json
import re
import random
import numpy as np
from sentence_transformers import SentenceTransformer

# === Load QA Data ===
@st.cache_data
def load_qa_data(path="data/crescent_qa.json"):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    questions = [item["question"] for item in data]
    answers = [item["answer"] for item in data]
    return questions, answers

# === Load Sentence Transformer Model ===
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

# === Compute and Cache Embeddings ===
@st.cache_data
def get_question_embeddings(questions, model):
    return model.encode(questions)

# === Top-k Search with Precomputed Embeddings ===
def get_top_k_matches(query, question_embeddings, questions, answers, model, k=3):
    query_embedding = model.encode([query])[0]
    scores = np.dot(question_embeddings, query_embedding) / (
        np.linalg.norm(question_embeddings, axis=1) * np.linalg.norm(query_embedding)
    )
    top_k_idx = np.argsort(scores)[::-1][:k]
    return [{"question": questions[i], "answer": answers[i], "score": float(scores[i])} for i in top_k_idx]

# === Input Preprocessing ===
def normalize_input(text):
    text = text.lower().strip()
    replacements = {
        "ur": "your", "u": "you", "pls": "please", "asap": "urgent", "wat": "what",
        "dept": "department", "schl": "school", "sme": "same"
    }
    for k, v in replacements.items():
        text = re.sub(rf"\b{k}\b", v, text)
    return text

# === Memory Handling ===
def update_memory(session, key, value):
    session[key] = value

def get_last_context(session, key):
    return session.get(key, None)

# === Tone Detection ===
def detect_tone(text):
    text = text.lower()
    if any(word in text for word in ["pls", "please", "hi", "hello", "thank", "good morning", "good afternoon"]):
        return "polite"
    if any(word in text for word in ["urgent", "now", "quick", "fast", "immediately", "asap"]):
        return "urgent"
    if any(word in text for word in ["why", "what", "how", "when", "confused", "help", "not sure", "don't understand"]):
        return "confused"
    if any(word in text for word in ["angry", "nonsense", "rubbish", "dumb", "idiot", "annoyed", "useless"]):
        return "angry"
    if re.search(r"[!?]{2,}", text):
        return "emphatic"
    return "neutral"

def rewrite_with_tone(user_input, response):
    tone = detect_tone(user_input)
    if tone == "polite":
        return "Sure! 😊 " + response
    elif tone == "urgent":
        return "Got it — here's the information you need right away:\n\n" + response
    elif tone == "confused":
        return "No worries, let me explain clearly:\n\n" + response
    elif tone == "angry":
        return "I'm here to help — let's sort this out calmly:\n\n" + response
    elif tone == "emphatic":
        return "Absolutely! Here's everything you need:\n\n" + response
    else:
        return "Here's what I found for you:\n\n" + response

# === Greeting Handling ===
def detect_greeting(text):
    greetings = ["hello", "hi", "hey", "good morning", "good afternoon", "good evening"]
    return any(greet in text.lower() for greet in greetings)

def detect_farewell(text):
    farewells = ["bye", "goodbye", "see you", "take care"]
    return any(farewell in text.lower() for farewell in farewells)

def get_random_greeting():
    return random.choice([
        "Hello! 😊 How can I assist you today?",
        "Hi there! 👋 What would you like to know?",
        "Hey! I'm here to help. Ask me anything about Crescent University."
    ])

# === Handle Input Logic ===
def handle_input(user_input, questions, answers, model, question_embeddings):
    user_input_norm = normalize_input(user_input)

    # Check for greeting or farewell
    if detect_greeting(user_input):
        return get_random_greeting()
    if detect_farewell(user_input):
        return "Goodbye! Feel free to come back with more questions anytime. 👋"

    # Search
    matches = get_top_k_matches(user_input_norm, question_embeddings, questions, answers, model)

    # Confidence logic
    top_match = matches[0]
    if top_match["score"] > 0.75:
        return rewrite_with_tone(user_input, top_match["answer"])
    else:
        return rewrite_with_tone(user_input, "I'm not entirely sure, but here's what I found:\n\n" + top_match["answer"])

# === Streamlit UI ===
st.set_page_config(page_title="CrescentBot", page_icon="🤖")
st.title("🤖 Crescent University Chatbot")
st.markdown("Ask me anything about departments, courses, or admission!")

# Session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Load data
questions, answers = load_qa_data()
model = load_model()
question_embeddings = get_question_embeddings(questions, model)

# User input
user_input = st.text_input("You:", key="user_input")
if user_input:
    response = handle_input(user_input, questions, answers, model, question_embeddings)
    st.session_state.chat_history.append(("You", user_input))
    st.session_state.chat_history.append(("Bot", response))
    st.session_state.user_input = ""  # clear input box

# Display history
for sender, msg in st.session_state.chat_history:
    if sender == "You":
        st.markdown(f"**🧑‍💬 You:** {msg}")
    else:
        st.markdown(f"**🤖 Bot:** {msg}")
