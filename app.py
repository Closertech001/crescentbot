import streamlit as st
import json
import random
import re
from openai import OpenAI
from utils.course_query import get_course_info
from utils.embedding import load_qa_data, get_top_k_matches
from utils.memory import update_memory, get_last_context
from utils.preprocess import normalize_input

# 🔐 OpenAI client
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# 📘 Load QA data and precomputed embeddings
qa_data, qa_embeddings = load_qa_data()

# 🤖 Load greeting/small talk logic
greeting_patterns = r"\b(hi|hello|hey|good morning|good afternoon|good evening|greetings)\b"
small_talk_phrases = [
    "I'm here if you need anything!",
    "What can I help you with today?",
    "Ask me anything about Crescent University.",
    "How can I assist you today?"
]

# 🧠 GPT fallback
def ask_gpt(query):
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": query}],
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()

# 🤝 Detect greetings
def detect_greeting(text):
    return re.search(greeting_patterns, text.lower()) is not None

# 💬 Main handler
def handle_input(user_input):
    user_input_norm = normalize_input(user_input)

    # Greeting
    if detect_greeting(user_input_norm):
        return random.choice(["Hello!", "Hi there!", "Greetings!", "Hey! 👋"])

    # Course info
    course_response = get_course_info(user_input_norm)
    if course_response:
        return course_response

    # Follow-up context
    last_context = get_last_context()
    if last_context:
        updated_input = f"{last_context} {user_input_norm}"
    else:
        updated_input = user_input_norm
    update_memory(updated_input)

    # Semantic match
    top_matches = get_top_k_matches(updated_input, qa_data, qa_embeddings, k=3)
    if top_matches and top_matches[0]["score"] > 0.7:
        return top_matches[0]["answer"]

    # GPT fallback
    return ask_gpt(user_input)

# 🎨 UI setup
st.set_page_config(page_title="Crescent University Chatbot", page_icon="🎓")
st.markdown("<h1 style='text-align:center;'>🎓 Crescent University Assistant</h1>", unsafe_allow_html=True)

# 💬 Input field
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Input container
with st.container():
    user_input = st.text_input("You:", key="user_input", placeholder="Ask me anything about Crescent University...")

# Handle new message
if user_input:
    response = handle_input(user_input)
    st.session_state.chat_history.append(("You", user_input))
    st.session_state.chat_history.append(("Bot", response))
    st.experimental_rerun()

# Display chat history
for sender, msg in st.session_state.chat_history:
    st.markdown(f"**{sender}:** {msg}")

# Typing simulation (optional)
if st.session_state.chat_history and st.session_state.chat_history[-1][0] == "You":
    with st.spinner("Bot is typing..."):
        pass
