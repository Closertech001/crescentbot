import streamlit as st
from utils.embedding import load_qa_embeddings
from utils.search import find_best_match
from utils.course_query import get_course_info
from utils.preprocess import normalize_input
from utils.greetings import is_greeting, greeting_responses, is_small_talk, small_talk_responses
from utils.memory import Memory
from openai import OpenAI
import time
import random

# --- LOAD MODELS AND DATA ---
qa_data, qa_embeddings = load_qa_embeddings()
memory = Memory()
client = OpenAI()

# --- STREAMLIT UI CONFIG ---
st.set_page_config(page_title="Crescent University Chatbot", page_icon="🎓")
st.markdown("<h2 style='text-align: center;'>Crescent University Chatbot 🤖🎓</h2>", unsafe_allow_html=True)

# Custom CSS
st.markdown("""
    <style>
    .message {padding: 8px; margin-bottom: 10px; border-radius: 15px;}
    .user-msg {background-color: #DCF8C6; text-align: right;}
    .bot-msg {background-color: #F1F0F0; text-align: left;}
    .typing {font-style: italic; color: gray;}
    </style>
""", unsafe_allow_html=True)

# --- SESSION STATE ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- FUNCTIONS ---

def display_message(message, sender="bot"):
    css_class = "bot-msg" if sender == "bot" else "user-msg"
    st.markdown(f"<div class='message {css_class}'>{message}</div>", unsafe_allow_html=True)

def bot_typing():
    with st.spinner("Bot is typing..."):
        time.sleep(random.uniform(0.5, 1.2))

def call_gpt_fallback(user_input):
    messages = [{"role": "user", "content": user_input}]
    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        temperature=0.5
    )
    return response.choices[0].message.content.strip()

def handle_input(user_input):
    normalized_input = normalize_input(user_input)
    memory.update_context(normalized_input)

    # GREETING
    if is_greeting(normalized_input):
        return random.choice(greeting_responses(normalized_input))

    # SMALL TALK
    if is_small_talk(normalized_input):
        return random.choice(small_talk_responses(normalized_input))

    # COURSE QUERY
    course_info = get_course_info(normalized_input)
    if course_info:
        return course_info

    # SEMANTIC SEARCH
    match = find_best_match(normalized_input, qa_data, qa_embeddings)
    if match:
        return match

    # FALLBACK TO GPT
    return call_gpt_fallback(user_input)

# --- MAIN APP ---

user_input = st.text_input("Ask me anything about Crescent University:", key="input")

if user_input:
    display_message(user_input, sender="user")
    bot_typing()
    response = handle_input(user_input)
    display_message(response, sender="bot")
    st.session_state.chat_history.append((user_input, response))

    # --- GREETING OR SMALL TALK ---
    if is_greeting(normalized_input):
        response = get_greeting_response()

    elif is_small_talk(normalized_input):
        response = get_small_talk_response(normalized_input)

    else:
        # --- COURSE OR DEPARTMENT QUERY ---
        course_answer = handle_course_query(normalized_input, memory)
        if course_answer:
            response = course_answer

        else:
            # --- SEMANTIC SEARCH ---
            match = semantic_search(normalized_input, qa_data, qa_embeddings)
            if match:
                response = match
            else:
                # --- FALLBACK TO GPT ---
                typing_animation()
                rewritten = rewrite_question(normalized_input)
                gpt_reply = client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant for Crescent University."},
                        {"role": "user", "content": rewritten}
                    ],
                    temperature=0.7
                )
                response = gpt_reply.choices[0].message.content

    st.session_state.chat_history.append(("bot", response))

# Display chat
for role, msg in st.session_state.chat_history:
    if role == "user":
        st.chat_message("user").markdown(msg)
    else:
        st.chat_message("assistant").markdown(msg)
