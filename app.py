# app.py

import streamlit as st
import os
import json
import re
import time
import numpy as np
import faiss
import openai
from sentence_transformers import SentenceTransformer
from textblob import TextBlob
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# ------------------- Inlined utils/embedding.py -------------------
def load_model(model_name="all-MiniLM-L6-v2"):
    return SentenceTransformer(model_name)

def load_dataset(filepath="data/crescent_qa.json"):
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    questions = [entry["question"] for entry in data]
    return data, questions

def get_question_embeddings(questions, model):
    return model.encode(questions, convert_to_numpy=True, normalize_embeddings=True)

def build_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    return index

# ------------------- Inlined utils/semantic_search.py -------------------
def search_semantic(query, model, index, questions, top_k=1):
    query_embedding = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    D, I = index.search(query_embedding, top_k)
    if I[0][0] == -1:
        return None, 0.0
    return questions[I[0][0]], float(D[0][0])

# ------------------- Inlined utils/preprocess.py -------------------
def normalize_query(query):
    return query.strip().lower()

def correct_typos(query):
    corrected = str(TextBlob(query).correct())
    return corrected if corrected != query else query

# ------------------- Inlined utils/greetings.py -------------------
import random
def get_smalltalk_response():
    responses = [
        "Sure thing! Here's what I found 👇",
        "Let me get that info for you 📘",
        "Alright, here’s what I know 🤖",
        "Absolutely! Let me explain 👇"
    ]
    return random.choice(responses)

# ------------------- Inlined utils/tone.py -------------------
def detect_tone(query):
    if any(w in query.lower() for w in ["please", "could you", "kindly"]):
        return "formal"
    elif any(w in query.lower() for w in ["lol", "😂", "btw", "hey", "yo"]):
        return "casual"
    elif any(w in query.lower() for w in ["joke", "funny", "humor"]):
        return "humorous"
    return "neutral"

def format_tone_response(tone, response):
    if tone == "formal":
        return f"Certainly. {response}"
    elif tone == "casual":
        return f"Cool! 😎 {response}"
    elif tone == "humorous":
        return f"Here's something to tickle your brain 🧠😄: {response}"
    return response

# ------------------- Inlined utils/rewrite.py -------------------
def rephrase_query_if_needed(query, known_questions, threshold=0.75):
    from difflib import SequenceMatcher
    best_match = None
    best_score = 0
    for q in known_questions:
        score = SequenceMatcher(None, query.lower(), q.lower()).ratio()
        if score > best_score:
            best_score = score
            best_match = q
    return best_match if best_score > threshold else query

# ------------------- Inlined utils/log_utils.py -------------------
def log_query(user_query, matched_question, score):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with open("logs.txt", "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] User: {user_query} | Match: {matched_question} | Score: {score:.2f}\n")

# ------------------- Inlined utils/memory.py -------------------
def init_memory():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

def update_memory(user_msg, bot_msg):
    st.session_state.chat_history.append({"user": user_msg, "bot": bot_msg})

def get_memory():
    return st.session_state.chat_history[-3:] if len(st.session_state.chat_history) >= 3 else st.session_state.chat_history

# ------------------- Inlined utils/course_query.py -------------------
def extract_course_query(query, course_data):
    query = query.lower()
    found = None
    for course in course_data:
        name = course["course_title"].lower()
        if name in query or course["course_code"].lower() in query:
            found = course
            break
    return found

# ------------------- Load Everything -------------------
model = load_model()
qa_data, questions = load_dataset("data/crescent_qa.json")
course_data = json.load(open("data/course_data.json", "r", encoding="utf-8"))
question_embeddings = get_question_embeddings(questions, model)
faiss_index = build_faiss_index(question_embeddings)
init_memory()

# ------------------- Streamlit UI -------------------
st.set_page_config(page_title="CrescentBot 🎓", page_icon="🤖")
st.markdown("<h1 style='text-align: center;'>🎓 CrescentBot – University Assistant</h1>", unsafe_allow_html=True)

with st.chat_message("assistant"):
    st.markdown("Hi there! Ask me anything about Crescent University. 💬")

# ------------------- Handle User Input -------------------
user_query = st.chat_input("Type your question here...")
if user_query:
    with st.spinner("Typing..."):
        norm_query = normalize_query(user_query)
        corrected_query = correct_typos(norm_query)
        rewritten_query = rephrase_query_if_needed(corrected_query, questions)
        tone = detect_tone(user_query)

        # First check for course info
        course = extract_course_query(user_query, course_data)
        if course:
            response = f"📘 *{course['course_code']} – {course['course_title']}*\n\n**Level**: {course['level']}\n**Semester**: {course['semester']}\n**Department**: {course['department']}\n**Faculty**: {course['faculty']}"
        else:
            matched_question, score = search_semantic(rewritten_query, model, faiss_index, questions)
            log_query(user_query, matched_question, score)
            if score > 0.7:
                answer = next((item["answer"] for item in qa_data if item["question"] == matched_question), "Sorry, I don't have an answer to that yet.")
                response = f"{get_smalltalk_response()}\n\n{answer}"
            else:
                fallback_prompt = f"You are CrescentBot, a helpful university assistant. Answer clearly and conversationally.\nUser: {user_query}"
                try:
                    completion = openai.ChatCompletion.create(
                        model="gpt-4",
                        messages=[{"role": "user", "content": fallback_prompt}],
                        temperature=0.7,
                        max_tokens=300
                    )
                    response = completion.choices[0].message.content.strip()
                except Exception as e:
                    response = "Sorry, I couldn't reach my brain (OpenAI) right now. Please try again later."

        # Tone & memory
        styled_response = format_tone_response(tone, response)
        update_memory(user_query, styled_response)

        # Display chat history
        for chat in get_memory():
            st.chat_message("user").markdown(chat["user"])
            st.chat_message("assistant").markdown(chat["bot"])

        # Clear input
        st.experimental_rerun()
