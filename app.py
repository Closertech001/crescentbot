import streamlit as st
import openai
import os
from dotenv import load_dotenv
from utils.embedding import load_model, load_dataset, get_question_embeddings, build_faiss_index
from utils.search import semantic_search_faiss
from utils.greetings import get_greeting
from utils.memory import update_memory, get_last_context
from datetime import datetime

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load model and data
@st.cache_resource
def initialize():
    model = load_model()
    qa_data, questions = load_dataset("data/qa_dataset.json")
    embeddings = get_question_embeddings(questions, model)
    index = build_faiss_index(embeddings)
    return model, index, qa_data, questions

model, index, qa_data, questions = initialize()

# --- UI Styling ---
st.set_page_config(page_title="CrescentBot 🎓", page_icon="🤖", layout="centered")
st.markdown("<h1 style='text-align: center;'>🤖 CrescentBot</h1>", unsafe_allow_html=True)
st.markdown("Ask me anything about Crescent University!")

# --- Session Memory ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Chat Input ---
user_input = st.chat_input("Ask something about Crescent University...")

# --- Show Greeting if No Chat Yet ---
if len(st.session_state.chat_history) == 0:
    greeting = get_greeting()
    st.chat_message("assistant").markdown(f"{greeting} 👋\n\nI'm CrescentBot! Ask me anything about the university.")

# --- Response Logic ---
if user_input:
    st.chat_message("user").markdown(user_input)

    # Check memory for context (optional for future features)
    prev_context = get_last_context(st.session_state.chat_history)

    # Semantic search
    matches = semantic_search_faiss(user_input, model, index, qa_data, questions, top_k=3)

    # Pick top answer
    best_match = matches[0] if matches else None

    if best_match and best_match["score"] > 0.5:
        response = f"📘 **Answer:** {best_match['answer']}"
        if best_match["department"]:
            response += f"\n\n🏛️ *Department:* {best_match['department']}"
    else:
        response = "🤔 Sorry, I couldn't find an exact answer. Could you try rephrasing?"

    st.chat_message("assistant").markdown(response)

    # Save memory
    update_memory(st.session_state.chat_history, user_input, response)
