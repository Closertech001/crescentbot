import streamlit as st
import time
import json
import random
from openai import OpenAI
from utils.embedding import load_qa_data, load_model, get_top_k_matches
from utils.course_query import get_course_info
from utils.greetings import (
    is_greeting, greeting_responses, is_small_talk, small_talk_response,
    extract_course_code, get_course_by_code, fallback_course_response
)
from utils.preprocess import normalize_input
from utils.memory import update_memory, get_last_context

# --- Streamlit Page Setup ---
st.set_page_config(page_title="Crescent University Chatbot", layout="wide")

st.markdown("""
    <style>
    .bot-typing {
        font-style: italic;
        color: gray;
        animation: blink 1s steps(5, start) infinite;
    }
    @keyframes blink {
        to { visibility: hidden; }
    }
    .chat-message { padding: 5px; }
    </style>
""", unsafe_allow_html=True)

# --- Load Models & Data ---
model = load_model()
qa_data, qa_embeddings = load_qa_data()
with open("data/course_data.json") as f:
    course_data = json.load(f)

# --- OpenAI Client ---
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# --- Memory ---
if "history" not in st.session_state:
    st.session_state.history = []
if "memory" not in st.session_state:
    st.session_state.memory = {}

# --- Sidebar ---
with st.sidebar:
    st.title("🎓 Crescent University Chatbot")
    st.markdown("Ask anything about departments, courses, fees, etc.")
    if st.button("🧹 Clear Chat"):
        st.session_state.history = []
        st.session_state.memory = {}

# --- Chat Interface ---
st.title("🤖 Ask me about Crescent University")
for msg in st.session_state.history:
    st.chat_message(msg["role"]).write(msg["content"])

user_input = st.chat_input("Ask a question...")

# --- Main Logic ---
if user_input:
    st.session_state.history.append({"role": "user", "content": user_input})
    user_input_norm = normalize_input(user_input)

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.empty():
            for dots in ["▌", "▌▌", "▌▌▌"]:
                st.markdown(f"<span class='bot-typing'>Bot is typing{dots}</span>", unsafe_allow_html=True)
                time.sleep(0.3)

        # 1. Handle greetings
        if is_greeting(user_input_norm):
            reply = greeting_responses(user_input_norm)

        # 2. Handle small talk
        elif is_small_talk(user_input_norm):
            reply = small_talk_response(user_input_norm)

        # 3. Handle course code lookup
        elif course_code := extract_course_code(user_input_norm):
            reply = get_course_by_code(course_code, course_data) or fallback_course_response(course_code)

        # 4. Handle department-level queries
        else:
            course_response = get_course_info(user_input_norm, course_data)
            if course_response:
                update_memory(st.session_state.memory, user_input_norm)
                reply = course_response
            else:
                # 5. Semantic search fallback
                top_match = get_top_k_matches(user_input_norm, qa_data, qa_embeddings, model, k=1)
                if top_match:
                    reply = random.choice([
                        "Here’s what I found for you:",
                        "This might help:",
                        "I think this is what you're looking for:",
                        "Based on my info, here you go:"
                    ]) + "\n\n" + top_match
                else:
                    # 6. Final GPT fallback
                    completion = client.chat.completions.create(
                        model="gpt-4",
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant for Crescent University."},
                            {"role": "user", "content": user_input}
                        ]
                    )
                    reply = completion.choices[0].message.content

        st.markdown(reply)
        st.session_state.history.append({"role": "assistant", "content": reply})
