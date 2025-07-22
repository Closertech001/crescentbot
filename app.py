# app.py

import streamlit as st
import random
import time
from openai import OpenAI
from utils.embedding import load_qa_dataset
from utils.course_query import get_course_info
from utils.greetings import is_greeting, greeting_responses, small_talk_response, detect_farewell
from utils.preprocess import normalize_input
from utils.memory import MemoryHandler
from utils.rewrite import rewrite_with_tone
from utils.search import search_similar_question
from utils.log_utils import log_query

# --- PAGE CONFIG ---
st.set_page_config(page_title="Crescent University Chatbot 🤖", page_icon="🎓")

# --- OPENAI CLIENT ---
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# --- LOAD DATA ---
qa_data, question_embeddings = load_qa_dataset()

# --- SESSION STATE ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

memory = MemoryHandler()

# --- STYLING ---
st.markdown(
    """
    <style>
    .message-bubble {
        padding: 0.6em 1em;
        border-radius: 1.2em;
        margin-bottom: 0.5em;
        display: inline-block;
        max-width: 85%;
    }
    .user {background-color: #DCF8C6; align-self: flex-end;}
    .bot {background-color: #F1F0F0;}
    .typing {
        font-style: italic;
        opacity: 0.6;
        margin-left: 0.5em;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🤖 Crescent University Chatbot")

# --- TYPING ANIMATION ---
def bot_typing_animation():
    with st.empty():
        for dots in ["", ".", "..", "..."]:
            st.markdown(f'<div class="typing">Bot dey reason{dots}</div>', unsafe_allow_html=True)
            time.sleep(0.4)

# --- DISPLAY CHAT HISTORY ---
for speaker, msg in st.session_state.chat_history:
    bubble_class = "user" if speaker == "You" else "bot"
    st.markdown(f'<div class="message-bubble {bubble_class}"><b>{speaker}:</b> {msg}</div>', unsafe_allow_html=True)

# --- HANDLE USER INPUT ---
def handle_input(user_input):
    normalized = normalize_input(user_input)

    if is_greeting(normalized):
        return greeting_responses(normalized)

    if detect_farewell(normalized):
        return "Bye bye! Hope say you go come back soon. 👋"

    small_talk = small_talk_response(normalized)
    if small_talk:
        return small_talk

    # Try course info
    course_result = get_course_info(normalized, memory)
    if course_result:
        log_query(normalized, score=1.0)
        return course_result

    # Try semantic search
    match = search_similar_question(normalized, qa_data, question_embeddings)
    if match and match["score"] > 0.7:
        log_query(normalized, score=match["score"])
        return random.choice([
            "Here's wetin I find for you 👇",
            "Hope this help you 😊",
            "Check this out 👇",
            "This fit help you: 🔍"
        ]) + "\n\n" + match["answer"]

    # GPT fallback
    log_query(normalized, score=0.0)
    messages = [
        {
            "role": "system",
            "content": "You be helpful assistant for Crescent University Nigeria. Answer all questions clearly based on the university's programs, departments, and student life. Use friendly tone and Pidgin when appropriate."
        },
        {"role": "user", "content": normalized}
    ]
    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages
    ).choices[0].message.content.strip()

    return rewrite_with_tone(user_input, response)

# --- USER INPUT ---
user_input = st.text_input("Ask me anything about the university...")

if user_input:
    st.session_state.chat_history.append(("You", user_input))
    bot_typing_animation()
    response = handle_input(user_input)
    st.session_state.chat_history.append(("Bot", response))
    st.experimental_rerun()  # Clear input
# app.py

import streamlit as st
import random
import time
from utils.embedding import load_qa_dataset, embed_and_search
from utils.course_query import get_course_info
from utils.greetings import is_greeting, greeting_responses, small_talk_response, detect_farewell
from utils.preprocess import normalize_input
from utils.memory import MemoryHandler
from utils.rewrite import rewrite_question
from utils.search import search_similar_question
from openai import OpenAI

# --- PAGE CONFIG ---
st.set_page_config(page_title="Crescent University Chatbot 🤖", page_icon="🎓")

# --- LOAD OPENAI ---
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# --- LOAD DATA ---
qa_data, question_embeddings = load_qa_dataset()

# --- SESSION STATE ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

memory = MemoryHandler()

# --- STYLING ---
st.markdown(
    """
    <style>
    .message-bubble {
        padding: 0.6em 1em;
        border-radius: 1.2em;
        margin-bottom: 0.5em;
        display: inline-block;
        max-width: 85%;
    }
    .user {background-color: #DCF8C6; align-self: flex-end;}
    .bot {background-color: #F1F0F0;}
    .typing {
        font-style: italic;
        opacity: 0.6;
        margin-left: 0.5em;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🤖 Crescent University Chatbot")

# --- DISPLAY CHAT HISTORY ---
for speaker, msg in st.session_state.chat_history:
    bubble_class = "user" if speaker == "You" else "bot"
    st.markdown(f'<div class="message-bubble {bubble_class}"><b>{speaker}:</b> {msg}</div>', unsafe_allow_html=True)

# --- TYPING ANIMATION ---
def bot_typing_animation():
    with st.empty():
        for dots in ["", ".", "..", "..."]:
            st.markdown(f'<div class="typing">Bot is typing{dots}</div>', unsafe_allow_html=True)
            time.sleep(0.4)

# --- HANDLE USER INPUT ---
def handle_input(user_input):
    normalized = normalize_input(user_input)

    if is_greeting(normalized):
        return greeting_responses(normalized)

    if detect_farewell(normalized):
        return "Bye bye! Hope say you go come back soon. 👋"

    small_talk = small_talk_response(normalized)
    if small_talk:
        return small_talk

    # Course info check
    course_result = get_course_info(normalized, memory)
    if course_result:
        return course_result

    # Semantic search
    match = search_similar_question(normalized, qa_data, question_embeddings)
    if match:
        return random.choice([
            "Here's wetin I find for you 👇",
            "Hope this help you 😊",
            "Check this out 👇",
            "This fit help you: 🔍"
        ]) + "\n\n" + match["answer"]

    # GPT fallback
    rewritten = rewrite_question(normalized, memory)
    messages = [
        {"role": "system", "content": "You be helpful assistant for Crescent University Nigeria. Answer questions based on the university context."},
        {"role": "user", "content": rewritten}
    ]

    completion = client.chat.completions.create(
        model="gpt-4",
        messages=messages
    )
    return completion.choices[0].message.content.strip()

# --- USER INPUT ---
user_input = st.text_input("Ask me anything about the university...")

if user_input:
    response = handle_input(user_input)
    st.session_state.chat_history.append(("You", user_input))
    bot_typing_animation()
    st.session_state.chat_history.append(("Bot", response))
    st.experimental_rerun()  # Clear input and rerun
