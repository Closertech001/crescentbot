import streamlit as st
import json
import os
import re
import time
import faiss
import numpy as np
import openai
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from textblob import TextBlob

# --- Load environment variables ---
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# --- Local file paths ---
QA_FILE = "data/crescent_qa.json"
COURSE_FILE = "data/course_data.json"
MODEL_NAME = "all-MiniLM-L6-v2"

# ---------------- UTILITY FUNCTIONS ---------------- #

# Load & embed
def load_model(name=MODEL_NAME):
    return SentenceTransformer(name)

def load_dataset(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    questions = [entry["question"] for entry in data]
    return data, questions

def get_embeddings(questions, model):
    return model.encode(questions, convert_to_numpy=True, normalize_embeddings=True)

def build_index(embeddings):
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index

# Semantic Search
def search_semantic(query, model, data, questions, index, k=1):
    query_vec = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    scores, idxs = index.search(query_vec, k)
    top_idx = idxs[0][0]
    return data[top_idx], scores[0][0]

# Tone Detection
def detect_tone(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0.3:
        return "cheerful"
    elif polarity < -0.2:
        return "formal"
    else:
        return "casual"

# Rewrite with tone
def rewrite_response(base_response, tone):
    if tone == "cheerful":
        return f"😊 Absolutely! {base_response}"
    elif tone == "formal":
        return f"Certainly. {base_response}"
    else:
        return f"{base_response}"

# Course code extractor
def extract_course_code(query):
    match = re.search(r"\b([A-Z]{3,5}\s?\d{3})\b", query.upper())
    return match.group(1).replace(" ", "") if match else None

# GPT Fallback
def gpt_fallback(prompt):
    try:
        res = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5
        )
        return res.choices[0].message.content.strip()
    except:
        res = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5
        )
        return res.choices[0].message.content.strip()

# ---------------- UI ELEMENTS ---------------- #

def typewriter_text(text, speed=0.02):
    placeholder = st.empty()
    typed = ""
    for char in text:
        typed += char
        placeholder.markdown(f"🟢 **CrescentBot:** {typed}▌")
        time.sleep(speed)
    placeholder.markdown(f"🟢 **CrescentBot:** {typed}")

def user_bubble(text):
    st.markdown(f"""
    <div style='background-color:#dcf8c6;padding:10px;border-radius:10px;margin:5px 0;text-align:right;'>
        <strong>🧑 You:</strong><br>{text}
    </div>
    """, unsafe_allow_html=True)

def bot_bubble(text):
    st.markdown(f"""
    <div style='background-color:#f1f0f0;padding:10px;border-radius:10px;margin:5px 0;'>
        <strong>🤖 CrescentBot:</strong><br>{text}
    </div>
    """, unsafe_allow_html=True)

# ---------------- APP LOGIC ---------------- #

def main():
    st.set_page_config(page_title="CrescentBot 🎓", page_icon="🤖")
    st.markdown("<h2 style='text-align:center;'>🎓 Crescent University Chat Assistant</h2>", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

    if "history" not in st.session_state:
        st.session_state.history = []

    # Load model and data
    model = load_model()
    qa_data, qa_questions = load_dataset(QA_FILE)
    course_data, _ = load_dataset(COURSE_FILE)

    # Prepare FAISS
    qa_embeddings = get_embeddings(qa_questions, model)
    index = build_index(qa_embeddings)

    # User input
    with st.form("query_form", clear_on_submit=True):
        query = st.text_input("💬 Ask me anything about Crescent University:")
        submitted = st.form_submit_button("Send")

    if submitted and query:
        user_bubble(query)

        course_code = extract_course_code(query)
        if course_code:
            match = next((c for c in course_data if c.get("course_code", "").upper() == course_code), None)
            if match:
                response = f"`{course_code}`: {match['course_title']} ({match.get('semester', '')}, Level {match.get('level', '')})"
            else:
                response = "❗ I couldn't find that course in our database."
        else:
            best_match, score = search_semantic(query, model, qa_data, qa_questions, index)
            response = best_match["answer"] if score > 0.6 else gpt_fallback(query)

        tone = detect_tone(query)
        final_response = rewrite_response(response, tone)

        # Show typing
        typewriter_text(final_response)

        # Append to history
        st.session_state.history.append({"user": query, "bot": final_response})

    # Show chat history
    if st.session_state.history:
        st.markdown("### 💬 Conversation History")
        for msg in st.session_state.history:
            user_bubble(msg["user"])
            bot_bubble(msg["bot"])

if __name__ == "__main__":
    main()
