import streamlit as st
import os
import json
import faiss
import numpy as np
import openai
from sentence_transformers import SentenceTransformer
from textblob import TextBlob
from dotenv import load_dotenv
from datetime import datetime
import re
import time

from utils.embedding import load_dataset, compute_question_embeddings
from utils.semantic_search import build_index, search
from utils.course_query import extract_course_query
from utils.text_utils import normalize_query
from utils.typo_corrector import correct_typos
from utils.department_detector import detect_department
from utils.logger import log_user_feedback

# --- Load environment variables ---
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# --- Load model without caching unhashables ---
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

# --- UI Styling ---
st.set_page_config(page_title="CrescentBot - University Assistant", layout="wide")
st.markdown(
    "<h2 style='text-align: center; color: #4B8BBE;'>🎓 Crescent University Chatbot</h2>", 
    unsafe_allow_html=True
)

# --- Memory store ---
if "history" not in st.session_state:
    st.session_state.history = []

# --- Greeting/Farewell logic ---
def detect_greeting_farewell(query):
    greetings = ["hello", "hi", "good morning", "good afternoon", "good evening", "hey"]
    thanks = ["thank you", "thanks", "thx", "thank u"]
    farewells = ["bye", "goodbye", "see you", "later"]

    norm = query.lower()
    if any(greet in norm for greet in greetings):
        hour = datetime.now().hour
        if hour < 12:
            return "🌅 Good morning! How can I assist you today?"
        elif hour < 17:
            return "🌞 Good afternoon! What would you like to know?"
        else:
            return "🌙 Good evening! Feel free to ask anything about Crescent University."
    elif any(thank in norm for thank in thanks):
        return "😊 You're welcome! Let me know if you need anything else."
    elif any(farewell in norm for farewell in farewells):
        return "👋 Goodbye! Have a great day!"
    return None

# --- GPT fallback ---
def gpt_fallback(query):
    try:
        res = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant for Crescent University. Only answer questions relevant to the university."},
                {"role": "user", "content": query}
            ],
            temperature=0.3
        )
        return res["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return "❌ Sorry, I'm currently unable to fetch a response from GPT-4."

# --- Main chatbot logic ---
def main():
    # Load data and model
    data = load_dataset("data/qa_dataset.json")
    model = load_model()
    embeddings = compute_question_embeddings(data, model)
    index = build_index(embeddings)

    st.markdown("##### Ask CrescentBot anything about the university below 👇")

    user_input = st.chat_input("Ask me a question...")

    if user_input:
        st.session_state.history.append({"role": "user", "content": user_input})
        norm_query = normalize_query(correct_typos(user_input))

        # Handle greetings/thanks/farewell directly
        quick_response = detect_greeting_farewell(norm_query)
        if quick_response:
            st.session_state.history.append({"role": "assistant", "content": quick_response})
        else:
            # Semantic search
            top_match, score = search(norm_query, index, model, data, top_k=1)
            course_code = extract_course_query(user_input)
            department = detect_department(norm_query)

            if score > 0.6:
                response = f"📘 *Here’s the info for* `{course_code}`:\n\n{top_match['answer']}" if course_code else top_match["answer"]
            else:
                response = gpt_fallback(norm_query)

            # Save and show response
            st.session_state.history.append({"role": "assistant", "content": response})

    # Display chat history
    for chat in st.session_state.history:
        if chat["role"] == "user":
            st.chat_message("user").markdown(f"👤 **You**: {chat['content']}")
        else:
            st.chat_message("assistant").markdown(f"🤖 **CrescentBot**: {chat['content']}")

# --- Run the app ---
if __name__ == "__main__":
    main()
