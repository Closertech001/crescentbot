import streamlit as st
import time
import random
import re

from utils.preprocess import preprocess_input
from utils.course_query import get_course_info
from utils.embedding import get_top_k_answers
from utils.search import semantic_search
from utils.memory import store_context_from_query, enrich_query_with_context
from utils.greetings import is_greeting, greeting_responses

from openai import OpenAI
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])


# --- UI CONFIG ---
st.set_page_config(page_title="Crescent University Bot", layout="centered")
st.markdown('<h1 style="text-align:center;">🎓 Crescent University Chatbot</h1>', unsafe_allow_html=True)
st.markdown('<style>' + open("assets/style.css").read() + '</style>', unsafe_allow_html=True)


# --- SESSION STATE ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- ANIMATION ---
def typing_animation():
    with st.empty():
        for i in range(3):
            dots = "." * (i + 1)
            st.markdown(f"**Bot is typing{dots}**")
            time.sleep(0.4)


# --- UTILITIES ---
def display_message(sender, message):
    if sender == "user":
        st.markdown(f'<div class="user-bubble">{message}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="bot-bubble">{message}</div>', unsafe_allow_html=True)


def dynamic_intro():
    return random.choice([
        "Here's what I found for you 👇",
        "Take a look at this 🧠",
        "This might help 📘",
        "Hope this answers your question ✅",
        "Check this out 👀"
    ])


# --- MAIN CHAT HANDLER ---
def handle_chat(user_input):
    original_input = user_input
    user_input = preprocess_input(user_input)

    display_message("user", original_input)

    # 1. Greeting
    if is_greeting(user_input):
        bot_reply = greeting_responses(user_input)
        typing_animation()
        display_message("bot", bot_reply)
        return

    # 2. Course info (department/level)
    query_info = get_course_info(user_input)
    query_info = enrich_query_with_context(query_info)

    if query_info:
        store_context_from_query(query_info)
        typing_animation()
        display_message("bot", dynamic_intro())

        for course in query_info.get("courses", []):
            display_message("bot", f"**{course['code']}** - {course['title']} ({course['unit']} unit{'s' if course['unit'] > 1 else ''})")
        return

    # 3. Semantic search
    top_match = get_top_k_answers(user_input, k=1)
    if top_match:
        typing_animation()
        display_message("bot", dynamic_intro())
        display_message("bot", top_match[0])
        return

    # 4. GPT Fallback
    typing_animation()
    gpt_reply = fallback_to_gpt(original_input)
    display_message("bot", gpt_reply)


# --- FALLBACK TO GPT-4 ---
def fallback_to_gpt(prompt):
    system = (
        "You are a helpful AI assistant for Crescent University Abeokuta. "
        "If a user asks about departments, courses, or admission, answer based on general university structure in Nigeria. "
        "If a user says 'final year', assume 400 or 500 level depending on program type. "
        "Don't hallucinate if you don't know the answer. Keep responses concise and friendly."
    )
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return "Sorry, I couldn't process that right now. Please try again later."


# --- SIDEBAR ---
with st.sidebar:
    st.title("🔧 Options")
    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()


# --- CHAT INPUT ---
user_query = st.chat_input("Ask me anything about Crescent University...")
if user_query:
    handle_chat(user_query)
