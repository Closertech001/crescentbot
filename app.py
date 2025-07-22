import streamlit as st
import os
import openai
import random
import faiss
import numpy as np
import sqlite3
import json
import re
from datetime import datetime
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from textblob import TextBlob

# === Load env vars & API ===
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# === Load data & model ===
from utils.embedding import load_dataset, compute_question_embeddings
dataset = load_dataset("data/qa_dataset.json")
model = SentenceTransformer("all-MiniLM-L6-v2")
questions = [item["question"] for item in dataset]
embeddings = compute_question_embeddings(model, questions)
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# === SQLite for memory ===
conn = sqlite3.connect("chat_memory.db")
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS memory (timestamp TEXT, user_input TEXT, bot_response TEXT)''')
conn.commit()

# === GREETING + SMALL TALK ===
GREETING_PATTERNS = [r"\bhi\b", r"\bhello\b", r"\bhey\b", r"\bgood (morning|afternoon|evening)\b", r"\bwhat's up\b", r"\bhowdy\b", r"\byo\b", r"\bsup\b", r"\bgreetings\b", r"\bhow far\b", r"\bhow you dey\b"]
GREETING_RESPONSES = {
    "positive": [
        "Hey there! 😊 You're sounding great today. How can I assist you?",
        "Hi! 👋 I'm glad you're feeling good. What would you like to know?",
        "Hello! 🌟 Ready to explore Crescent University together?"
    ],
    "neutral": [
        "Hi there! 😊 How can I help you?",
        "Hello! 👋 What would you like to know about Crescent University?",
        "Hey! I'm here to assist you with your course or university questions.",
        "Hi! Let me know what you're looking for.",
        "How far! I'm here for any Crescent Uni gist you need."
    ],
    "negative": [
        "I'm here to help — let’s figure it out together. 💡",
        "Sorry if you're having a rough time. Let's fix that. What do you need?",
        "I’ve got your back. Let me help you with that. 💪"
    ]
}
SMALL_TALK_PATTERNS = {
    r"how are you": [
        "I'm doing great, thanks for asking! 😊 How can I help you today?",
        "Feeling sharp and ready to assist! ✨"
    ],
    r"who (are|created|made) you": [
        "I'm the Crescent University Chatbot 🤖, built to help students like you!",
        "I was created to guide you through Crescent Uni life 📘"
    ],
    r"what can you do": [
        "I can help you with course info, departments, fees, and more 🎓",
        "Ask me about admission, courses, or departments — I’ve got answers! 💡"
    ],
    r"tell me about yourself": [
        "I'm a smart little assistant for Crescent University 🧠💬",
        "I answer questions about courses, fees, staff, and more!"
    ],
    r"are you (smart|intelligent)": [
        "I try my best! 😄 Especially when it comes to university questions.",
        "Not bad for a chatbot, right? 😉"
    ],
    r"you('?| )re (funny|cool|smart)": [
        "Aww, thanks! 😊 You’re not so bad yourself.",
        "Appreciate it! Let’s keep the good vibes going 🔥"
    ]
}

def is_greeting(user_input):
    return any(re.search(p, user_input.lower()) for p in GREETING_PATTERNS)

def detect_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity
    return "positive" if polarity > 0.2 else "negative" if polarity < -0.2 else "neutral"

def greeting_response(text):
    sentiment = detect_sentiment(text)
    return random.choice(GREETING_RESPONSES[sentiment])

def is_small_talk(user_input):
    return any(re.search(p, user_input.lower()) for p in SMALL_TALK_PATTERNS)

def small_talk_response(user_input):
    for pattern, replies in SMALL_TALK_PATTERNS.items():
        if re.search(pattern, user_input.lower()):
            return random.choice(replies)
    return "I'm here for all your Crescent University questions! 🎓"

# === Course Code ===
def extract_course_code(text):
    match = re.search(r"\b([A-Z]{2,4})\s?(\d{3})\b", text.upper())
    return f"{match.group(1)} {match.group(2)}" if match else None

def get_course_by_code(code, course_data):
    code = code.upper().strip()
    for entry in course_data:
        if code in entry.get("answer", ""):
            parts = entry["answer"].split(" | ")
            for part in parts:
                if part.strip().startswith(code):
                    return part.strip()
    return None

# === GPT Fallback ===
def call_gpt_fallback(user_input):
    try:
        messages = [
            {"role": "system", "content": "You are CrescentBot, a smart assistant for Crescent University. Respond clearly and helpfully."},
            {"role": "user", "content": user_input}
        ]
        response = openai.ChatCompletion.create(model="gpt-4", messages=messages)
        return response.choices[0].message.content.strip()
    except Exception as e:
        return "😕 Sorry, I'm having trouble reaching my brain right now. Please try again shortly."

# === Semantic Search ===
def search_semantic(user_input, top_k=1):
    query_emb = model.encode([user_input])
    scores, indices = index.search(query_emb, top_k)
    match = dataset[indices[0][0]]["answer"]
    return match, scores[0][0]

# === Streamlit UI ===
st.set_page_config(page_title="Crescent University Chatbot", page_icon="🎓")
st.title("🎓 Crescent University Assistant")

if "history" not in st.session_state:
    st.session_state.history = []

with st.chat_message("assistant"):
    st.markdown("Hi there! I'm **CrescentBot** 🤖 — Ask me anything about Crescent University!")

user_input = st.chat_input("Type your question...")

if user_input:
    st.chat_message("user").markdown(user_input)

    # === Logic Chain ===
    if is_greeting(user_input):
        reply = greeting_response(user_input)

    elif is_small_talk(user_input):
        reply = small_talk_response(user_input)

    elif course_code := extract_course_code(user_input):
        course_info = get_course_by_code(course_code, dataset)
        reply = f"📘 *Here’s the info for* `{course_code}`:\n\n{course_info}" if course_info else "I couldn’t find details for that course code. 🤔"

    else:
        match, score = search_semantic(user_input)
        reply = match if score < 0.5 else call_gpt_fallback(user_input)

    # Display & store
    st.chat_message("assistant").markdown(reply)
    st.session_state.history.append((user_input, reply))
    c.execute("INSERT INTO memory VALUES (?, ?, ?)", (datetime.now(), user_input, reply))
    conn.commit()
