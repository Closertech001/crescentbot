import streamlit as st
import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from textblob import TextBlob
from symspellpy import SymSpell
from dotenv import load_dotenv
from datetime import datetime
import re
import time
from openai import OpenAI

# --- Load environment variables ---
load_dotenv()
client = OpenAI()

# --- Load datasets ---
with open("data/crescent_qa.json", "r") as f:
    crescent_data = json.load(f)

with open("data/course_data.json", "r") as f:
    course_data = json.load(f)

# --- Load model ---
def load_model(name="all-MiniLM-L6-v2"):
    return SentenceTransformer(name)

# --- Preprocess text ---
def normalize(text):
    return re.sub(r"[^a-zA-Z0-9\s]", "", text.lower()).strip()

# --- SymSpell setup ---
def setup_symspell():
    symspell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
    dict_path = "frequency_dictionary_en_82_765.txt"
    symspell.load_dictionary(dict_path, term_index=0, count_index=1)
    return symspell

symspell = setup_symspell()

# --- Typo correction ---
def correct_typo(text):
    suggestions = symspell.lookup_compound(text, max_edit_distance=2)
    return suggestions[0].term if suggestions else text

# --- Tone detection ---
def detect_tone(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0.3:
        return "😊"
    elif polarity < -0.3:
        return "😟"
    else:
        return "🙂"

# --- Semantic search ---
def build_index(questions, model):
    embeddings = model.encode(questions)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index, embeddings

def search_semantic(query, index, model, data, top_k=3):
    query_vec = model.encode([query])
    D, I = index.search(query_vec, top_k)
    results = [{"score": float(1 - D[0][i]), "answer": data[I[0][i]]["answer"]} for i in range(len(I[0]))]
    return results[0] if results else {"answer": None, "score": 0.0}

# --- GPT fallback ---
def gpt_fallback(query):
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful university assistant."},
                {"role": "user", "content": query}
            ],
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print("⚠️ GPT-4 failed, falling back to GPT-3.5:", e)
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful university assistant."},
                    {"role": "user", "content": query}
                ],
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print("❌ GPT-3.5 also failed:", e)
            return "😔 Sorry, I couldn't fetch a response at the moment. Please try again later."

# --- Course query detection ---
def extract_course_query(query):
    course_match = re.search(r"([a-zA-Z]{3,5}\s*\d{3})", query)
    if course_match:
        return course_match.group(1).replace(" ", "").upper()
    return None

def get_course_info(course_code):
    for course in course_data:
        if course.get("code", "").upper() == course_code.upper():
            return course
    return None

# --- Chat UI layout ---
st.set_page_config(page_title="🎓 Crescent University Chatbot", layout="wide")
st.markdown("""
    <style>
    .user-message {background-color: #e0f7fa; padding: 10px; border-radius: 10px; margin-bottom: 10px;}
    .bot-message {background-color: #fff3e0; padding: 10px; border-radius: 10px; margin-bottom: 10px;}
    </style>
""", unsafe_allow_html=True)

st.title("🤖 Crescent University Assistant")
st.write("Ask me anything about Crescent University ✨")

# --- Session memory ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Load model & build index ---
model = load_model()
questions = [normalize(q["question"]) for q in crescent_data]
index, _ = build_index(questions, model)

# --- Main interface ---
def main():
    query = st.text_input("💬 Your question:", key="user_input")
    if query:
        norm_query = normalize(correct_typo(query))
        tone_emoji = detect_tone(query)

        course_code = extract_course_query(norm_query)
        if course_code:
            course_info = get_course_info(course_code)
            if course_info:
                response = f"📘 *Here’s the info for* `{course_code}`:\n**Title:** {course_info['title']}\n**Units:** {course_info['unit']}\n**Semester:** {course_info['semester']}\n**Level:** {course_info['level']}"
            else:
                response = f"🔍 Sorry, I couldn't find any course with code `{course_code}`."
        else:
            best_match = search_semantic(norm_query, index, model, crescent_data)
            score = best_match["score"]
            response = best_match["answer"] if score > 0.6 else gpt_fallback(query)

        st.session_state.chat_history.append((query, response, tone_emoji))

    # --- Display chat ---
    for user_msg, bot_msg, tone in st.session_state.chat_history:
        st.markdown(f"<div class='user-message'>👤 <b>You:</b> {user_msg}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='bot-message'>{tone} <b>CrescentBot:</b> {bot_msg}</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
