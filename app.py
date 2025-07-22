import streamlit as st
import random
import re
import time

from utils.course_query import get_course_info
from utils.embedding import get_top_k_answer
from utils.search import semantic_search
from utils.memory import MemoryHandler
from utils.greetings import is_greeting, is_small_talk, respond_to_small_talk, get_greeting_response
from utils.preprocess import normalize_input
from openai import OpenAI

# --- Load OpenAI API key ---
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# --- Initialize memory handler ---
memory = MemoryHandler()

# --- Streamlit UI setup ---
st.set_page_config(page_title="Crescent University Chatbot", layout="wide")
st.markdown('<h1 style="text-align: center;">🎓 Crescent University Chatbot</h1>', unsafe_allow_html=True)

with st.sidebar:
    st.image("assets/cu_logo.png", width=200)
    if st.button("🧹 Clear Chat"):
        st.session_state.messages = []
        memory.clear()

# --- Initialize chat history ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Chatbot typing animation ---
def bot_typing():
    with st.spinner("Bot is typing..."):
        time.sleep(random.uniform(0.7, 1.3))

# --- Display chat messages ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Handle user input ---
user_input = st.chat_input("Ask me anything about Crescent University...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # --- Normalize input ---
    normalized_input = normalize_input(user_input)

    # --- Greeting ---
    if is_greeting(normalized_input):
        response = get_greeting_response()
        with st.chat_message("assistant"):
            bot_typing()
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.stop()

    # --- Small Talk ---
    if is_small_talk(normalized_input):
        response = respond_to_small_talk(normalized_input)
        with st.chat_message("assistant"):
            bot_typing()
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.stop()

    # --- Course / Department Info ---
    course_response = get_course_info(normalized_input, memory)
    if course_response:
        with st.chat_message("assistant"):
            bot_typing()
            st.markdown(course_response)
        st.session_state.messages.append({"role": "assistant", "content": course_response})
        st.stop()

    # --- Semantic Search ---
    result = semantic_search(normalized_input, top_k=3)
    if result:
        dynamic_openings = [
            "Here's something I found for you:",
            "This might help:",
            "Take a look at this:",
            "Hope this answers your question:",
            "Here's what I found:"
        ]
        response = f"**{random.choice(dynamic_openings)}**\n\n{result}"
        with st.chat_message("assistant"):
            bot_typing()
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.stop()

    # --- Fallback: Use GPT-4 ---
    try:
        with st.chat_message("assistant"):
            bot_typing()
            fallback_prompt = f"You are a helpful assistant for Crescent University. Answer this question clearly and concisely:\n\n{user_input}"
            completion = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "system", "content": fallback_prompt}],
                max_tokens=500
            )
            gpt_response = completion.choices[0].message.content.strip()
            st.markdown(gpt_response)
        st.session_state.messages.append({"role": "assistant", "content": gpt_response})
    except Exception as e:
        error_msg = "I'm sorry, I couldn't process that right now. Please try again later."
        with st.chat_message("assistant"):
            st.error(error_msg)
        st.session_state.messages.append({"role": "assistant", "content": error_msg})
