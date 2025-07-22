import streamlit as st
import os
import json
import faiss
import numpy as np
import openai
from sentence_transformers import SentenceTransformer
from textblob import TextBlob
from dotenv import load_dotenv
import re

# --- Load environment variables ---
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# --- Set page config ---
st.set_page_config(page_title="CrescentBot", page_icon="🤖", layout="centered")

# --- Load datasets ---
@st.cache_data
def load_dataset():
    try:
        with open("data/crescent_qa.json", "r", encoding="utf-8") as f:
            data = json.load(f)
            questions = [item["question"] for item in data]
            return data, questions
    except FileNotFoundError:
        st.warning("❌ 'crescent_qa.json' not found. Please upload the file to `data/`.")
        return [], []

@st.cache_data
def load_course_data():
    try:
        with open("data/course_data.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        st.warning("⚠️ 'course_data.json' not found. Some course-specific queries may not work.")
        return {}

# --- Embedding ---
def compute_question_embeddings(questions, model):
    return model.encode(questions, convert_to_tensor=False, normalize_embeddings=True)

# --- Semantic Search ---
def semantic_search(query, index, model, questions, qa_data, top_k=1):
    query_embedding = model.encode([query], convert_to_tensor=False, normalize_embeddings=True)
    D, I = index.search(np.array(query_embedding), top_k)
    top_idx = I[0][0]
    score = D[0][0]
    matched_q = questions[top_idx]
    matched_a = qa_data[top_idx]["answer"]
    return matched_a, matched_q, score

# --- GPT Fallback ---
def fallback_response(query):
    try:
        completion = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are CrescentBot, an assistant for Crescent University."},
                {"role": "user", "content": query},
            ],
            temperature=0.5,
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        return "❌ Sorry, I'm currently unable to fetch a response from GPT."

# --- Sentiment Detection ---
def detect_sentiment(text):
    analysis = TextBlob(text)
    return "positive" if analysis.sentiment.polarity > 0.1 else "negative" if analysis.sentiment.polarity < -0.1 else "neutral"

# --- Initialize Model + Index ---
@st.cache_resource
def initialize():
    qa_data, questions = load_dataset()
    course_data = load_course_data()

    if not qa_data or not questions:
        return None, None, qa_data, questions, course_data

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embeddings = compute_question_embeddings(questions, model)

    if len(embeddings) == 0:
        return None, None, qa_data, questions, course_data

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))

    return model, index, qa_data, questions, course_data

# --- UI ---
st.title("🤖 CrescentBot – Crescent University Assistant")
st.markdown("Ask anything about Crescent University — I’ll try my best to help! 🎓")

query = st.text_input("💬 Type your question here", placeholder="e.g., What are the 100 level courses in Anatomy?")

if query:
    model, index, qa_data, questions, course_data = initialize()

    if model is None or index is None or not qa_data:
        st.error("❌ Cannot respond: Knowledge base not loaded.")
        st.stop()

    sentiment = detect_sentiment(query)
    with st.spinner("Thinking..."):
        answer, matched_question, score = semantic_search(query, index, model, questions, qa_data)

        # If similarity score is poor, use GPT fallback
        if score > 50:
            response = fallback_response(query)
            st.markdown(f"💬 *GPT Response:*\n\n{response}")
        else:
            st.markdown(f"💡 *Matched:* `{matched_question}`")
            st.markdown(f"📘 {answer}")
