import streamlit as st
import json
import re
import time
import openai

# Custom utilities
from utils.embedding import load_qa_embeddings, find_most_similar_question
from utils.course_query import extract_course_info
from utils.preprocess import normalize_input
from utils.memory import MemoryHandler
from utils.greetings import detect_greeting, generate_greeting_response, is_small_talk, generate_small_talk_response
from utils.rewrite import rewrite_followup
from utils.search import fallback_to_gpt_if_needed
from utils.tone import detect_tone

# Set OpenAI API Key
try:
    openai.api_key = st.secrets["OPENAI_API_KEY"]
except KeyError:
    st.error("❌ OpenAI API key not found in secrets. Please set OPENAI_API_KEY in .streamlit/secrets.toml")
    st.stop()

# Cache QA data & embeddings for speed
@st.cache_data(show_spinner="Loading knowledge base...")
def load_data():
    return load_qa_embeddings("data/crescent_qa.json")

qa_data, question_embeddings = load_data()

# Initialize memory
memory = MemoryHandler()

# Confidence threshold
CONFIDENCE_THRESHOLD = 0.75


# 💬 Main Chatbot Function
def crescent_chatbot():
    st.set_page_config(page_title="Crescent University Chatbot", layout="centered")
    st.markdown("<h1 style='text-align: center;'>🎓 Crescent University Chatbot 🤖</h1>", unsafe_allow_html=True)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display past messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    user_input = st.chat_input("Ask me anything about Crescent University...")

    if user_input and user_input.strip():
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Let me check that for you..."):
                response = get_bot_response(user_input)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})


# 🤖 Bot Response Logic
def get_bot_response(user_input):
    normalized_input = normalize_input(user_input)
    tone = detect_tone(user_input)

    # Friendly pre-response tone reaction (optional)
    if tone == "angry":
        return "I'm here to help, no worries. Let’s solve this together. 😊"
    elif tone == "urgent":
        st.toast("✅ I’ll prioritize that for you!")

    # Rewrite follow-up question based on memory
    rewritten_input = rewrite_followup(normalized_input, memory)

    # Greeting
    if detect_greeting(rewritten_input):
        return generate_greeting_response()

    # Small talk
    if is_small_talk(rewritten_input):
        return generate_small_talk_response()

    # Course-specific info
    course_response = extract_course_info(rewritten_input, memory)
    if course_response:
        return course_response

    # Semantic QA match
    best_match, score = find_most_similar_question(rewritten_input, qa_data, question_embeddings)
    if score > CONFIDENCE_THRESHOLD:
        memory.update_last_topic(best_match)
        return best_match["answer"]

    # Fallback to GPT
    return fallback_to_gpt_if_needed(user_input)


# 🚀 Launch App
if __name__ == "__main__":
    crescent_chatbot()
