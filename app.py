# app.py - Crescent University Chatbot with Full Features

import streamlit as st
import json
import faiss
import os
import time
import numpy as np
from datetime import datetime
from sentence_transformers import SentenceTransformer
from textblob import TextBlob

# --- Inject CSS for chat styling ---
def inject_css():
    with open("assets/style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# --- Utils: Embedding ---
def load_model(name="all-MiniLM-L6-v2"):
    return SentenceTransformer(name)

def load_dataset(path="data/crescent_qa.json"):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    questions = [entry["question"] for entry in data]
    return data, questions

def get_question_embeddings(questions, model):
    return model.encode(questions, convert_to_numpy=True, normalize_embeddings=True)

def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index

# --- Utils: Semantic Search ---
def search_semantic(query, model, index, questions, data, top_k=1):
    q_embedding = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    scores, indices = index.search(q_embedding, top_k)
    result = []
    for i in indices[0]:
        result.append(data[i])
    return result[0] if result else {"answer": "Sorry, I couldn't find that."}

# --- Utils: Preprocessing ---
def normalize_text(text):
    return text.strip().lower()

def correct_spelling(text):
    blob = TextBlob(text)
    return str(blob.correct())

# --- Utils: Course Query Extraction ---
def extract_course_query(query):
    query = query.lower()
    keywords = ["100", "200", "300", "400", "first", "second", "semester", "level"]
    if any(k in query for k in keywords) and "course" in query:
        return query
    return None

def search_courses(query, course_file="data/course_data.json"):
    with open(course_file, "r", encoding="utf-8") as f:
        course_data = json.load(f)
    query = normalize_text(query)
    results = []
    for item in course_data:
        course_name = item.get("course_name", "").lower()
        course_code = item.get("course_code", "").lower()
        if course_name in query or course_code in query or str(item.get("level")) in query:
            results.append(item)
    return results

# --- Utils: Tone Detection ---
def detect_tone(user_input):
    input_lower = user_input.lower()
    if "😂" in user_input or "joke" in input_lower:
        return "humorous"
    elif any(w in input_lower for w in ["please", "kindly", "would you"]):
        return "formal"
    elif any(w in input_lower for w in ["yo", "sup", "hey bro", "what's up"]):
        return "casual"
    return "neutral"

def respond_with_tone(response, tone):
    if tone == "formal":
        return f"Certainly. {response}"
    elif tone == "casual":
        return f"Cool! 😎 {response}"
    elif tone == "humorous":
        return f"Alright, here's the scoop... 😂 {response}"
    else:
        return response

# --- Utils: Greetings ---
def detect_greeting(text):
    greetings = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"]
    return any(greet in text.lower() for greet in greetings)

def get_greeting_response():
    return "👋 Hello! How can I assist you with Crescent University today?"

# --- Chat UI Display ---
def display_chat(message, sender="bot"):
    css_class = "bot-message" if sender == "bot" else "user-message"
    st.markdown(f'<div class="chat-message {css_class}">{message}</div>', unsafe_allow_html=True)

# --- App Execution ---
def main():
    st.set_page_config("CrescentBot 🎓", page_icon="🤖", layout="centered")
    inject_css()

    st.title("🤖 CrescentBot — University Assistant")
    st.markdown("Ask me anything about Crescent University!")

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    model = load_model()
    data, questions = load_dataset()
    embeddings = get_question_embeddings(questions, model)
    index = build_faiss_index(embeddings)

    # Chat interface
    user_input = st.chat_input("Ask a question...")

    if user_input:
        tone = detect_tone(user_input)
        norm_query = normalize_text(correct_spelling(user_input))

        st.session_state.messages.append({"sender": "user", "text": user_input})
        display_chat(user_input, sender="user")

        with st.spinner("Typing..."):
            time.sleep(0.8)  # simulate typing delay
            if detect_greeting(norm_query):
                bot_reply = get_greeting_response()
            elif extract_course_query(norm_query):
                results = search_courses(norm_query)
                if results:
                    course_lines = "\n".join(
                        f"📘 **{c['course_code']}**: {c['course_name']}" for c in results
                    )
                    bot_reply = f"Here are the matching courses:\n\n{course_lines}"
                else:
                    bot_reply = "I couldn't find any matching courses."
            else:
                result = search_semantic(norm_query, model, index, questions, data)
                bot_reply = result["answer"]

            styled_reply = respond_with_tone(bot_reply, tone)
            st.session_state.messages.append({"sender": "bot", "text": styled_reply})
            display_chat(styled_reply, sender="bot")

    # Display message history
    for msg in st.session_state.messages:
        display_chat(msg["text"], sender=msg["sender"])

if __name__ == "__main__":
    main()
