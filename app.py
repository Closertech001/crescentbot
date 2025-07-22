import streamlit as st
import json
import os
import faiss
import numpy as np
import openai
from sentence_transformers import SentenceTransformer
from textblob import TextBlob
from dotenv import load_dotenv
import re

# Load .env
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# ---------- UTILS INLINE ----------

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_data
def load_dataset():
    with open("crescent_qa.json", "r", encoding="utf-8") as f:
        qa_data = json.load(f)
    questions = [item["question"] for item in qa_data]
    return qa_data, questions

@st.cache_resource
def get_question_embeddings(model, questions):
    return np.array(model.encode(questions, convert_to_numpy=True))

@st.cache_resource
def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

def semantic_search_faiss(query, model, index, questions, top_k=1):
    query_embedding = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)
    results = [(questions[i], distances[0][j]) for j, i in enumerate(indices[0])]
    return results

def fallback_gpt_response(user_query, context=None):
    messages = [{"role": "system", "content": "You are CrescentBot, an academic assistant for Crescent University."}]
    if context:
        messages.append({"role": "system", "content": f"Previous context: {context}"})
    messages.append({"role": "user", "content": user_query})

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages,
            temperature=0.4,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return "⚠️ Sorry, I'm currently unable to fetch a response. Please try again later."

def correct_spelling(text):
    return str(TextBlob(text).correct())

def is_plain_greeting(text):
    greetings = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"]
    return text.lower().strip() in greetings

# ---------- INIT ----------

@st.cache_resource
def initialize():
    model = load_model()
    qa_data, questions = load_dataset()
    question_embeddings = get_question_embeddings(model, questions)
    index = build_faiss_index(question_embeddings)
    return model, index, qa_data, questions

# ---------- UI ----------

st.set_page_config(page_title="CrescentBot 🎓", layout="wide")
st.markdown("<h2 style='text-align: center;'>🤖 CrescentBot – Crescent University Assistant</h2>", unsafe_allow_html=True)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_query = st.chat_input("Ask CrescentBot anything...")

if user_query:
    with st.spinner("Thinking..."):
        model, index, qa_data, questions = initialize()
        corrected_query = correct_spelling(user_query)

        if is_plain_greeting(corrected_query):
            bot_response = "👋 Hello! How can I assist you with Crescent University today?"
        else:
            results = semantic_search_faiss(corrected_query, model, index, questions, top_k=1)
            top_match, score = results[0]
            matched_entry = next((item for item in qa_data if item["question"] == top_match), None)

            if matched_entry and score < 1.0:
                bot_response = matched_entry["answer"]
            else:
                bot_response = fallback_gpt_response(user_query)

        # Save and display chat
        st.session_state.chat_history.append(("You", user_query))
        st.session_state.chat_history.append(("CrescentBot", bot_response))

# Display chat history
for sender, message in st.session_state.chat_history:
    with st.chat_message("user" if sender == "You" else "assistant"):
        st.markdown(message)
