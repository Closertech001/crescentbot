# app.py

import streamlit as st
import openai
import os
from dotenv import load_dotenv
from utils.embedding import load_model, load_qa_data, get_question_embeddings, build_faiss_index
from utils.search import semantic_search_faiss
from utils.greetings import get_greeting
from utils.memory import update_memory, get_last_context
from textblob import TextBlob

# --- Load environment variables ---
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# --- Initialize Session State ---
if "memory" not in st.session_state:
    st.session_state.memory = {}

# --- Load and embed data ---
@st.cache_resource
def initialize():
    model = load_model()
    qa_data = load_qa_data("data/crescent_qa.json")
    questions = [entry["question"] for entry in qa_data]
    embeddings = get_question_embeddings(questions, model)
    index = build_faiss_index(embeddings)
    return model, qa_data, questions, index

model, qa_data, questions, index = initialize()

# --- Streamlit UI ---
st.title("🎓 Crescent University Assistant")

user_input = st.chat_input("Ask me anything about Crescent University...")
if user_input:
    st.chat_message("user").markdown(user_input)

    # --- Greeting check ---
    lower_input = user_input.lower()
    if any(greet in lower_input for greet in ["hello", "hi", "hey", "good morning", "good evening", "good afternoon"]):
        greeting = get_greeting()
        st.chat_message("assistant").markdown(f"{greeting} 👋 How can I help you today?")
    else:
        # Semantic search
        matches = semantic_search_faiss(user_input, model, index, qa_data, questions, top_k=3)
        best_match = matches[0] if matches and matches[0]["score"] > 0.75 else None

        if best_match:
            response = best_match["answer"]
        else:
            # GPT fallback
            try:
                completion = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=[{"role": "system", "content": "You are a helpful assistant for Crescent University."},
                              {"role": "user", "content": user_input}]
                )
                response = completion.choices[0].message.content.strip()
            except Exception as e:
                response = "⚠️ Sorry, I couldn't fetch a response right now."

        st.chat_message("assistant").markdown(response)

        # --- Memory update ---
        update_memory(st.session_state.memory, "last_query", user_input)
        update_memory(st.session_state.memory, "last_response", response)
