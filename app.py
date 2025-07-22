import streamlit as st
import random
import time
import json
import re

from utils.course_query import get_course_info
from utils.embedding import load_embeddings
from utils.search import semantic_search
from utils.memory import MemoryHandler
from utils.greetings import is_greeting, is_small_talk, respond_to_small_talk
from utils.preprocess import normalize_input
from utils.rewrite import rewrite_question

import openai

# 🔐 Load OpenAI API key
openai.api_key = st.secrets["OPENAI_API_KEY"]

# 📦 Load course data and embeddings
with open("data/course_data.json", "r") as f:
    course_data = json.load(f)

qa_data, embeddings, model = load_embeddings("data/crescent_qa.json")

# 🧠 Initialize session memory
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "memory" not in st.session_state:
    st.session_state.memory = MemoryHandler()

# 🎨 Custom CSS
with open("assets/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title("🎓 Crescent University Chatbot")

# 💬 Chat message function
def add_message(role, content):
    with st.chat_message(role):
        st.markdown(content)
    st.session_state.chat_history.append({"role": role, "content": content})

# 🤖 Bot typing animation
def bot_typing():
    with st.chat_message("assistant"):
        dots = st.empty()
        for i in range(3):
            dots.markdown("Bot is typing" + "." * (i + 1))
            time.sleep(0.4)
        dots.empty()

# 🔁 GPT fallback
def gpt_fallback(question):
    messages = [{"role": "system", "content": "You are a helpful Crescent University assistant."}]
    for msg in st.session_state.chat_history[-5:]:
        messages.append(msg)
    messages.append({"role": "user", "content": question})

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages,
        temperature=0.7,
        max_tokens=500
    )
    return response.choices[0].message.content.strip()

# 🎯 Answer logic
def get_answer(user_input):
    norm_input = normalize_input(user_input)
    mem = st.session_state.memory

    # Greeting
    if is_greeting(norm_input):
        return random.choice([
            "Hello! How can I assist you today?",
            "Hi there 👋, what would you like to know?",
            "Hey! Feel free to ask me about Crescent University."
        ])

    # Small talk
    if is_small_talk(norm_input):
        return respond_to_small_talk(norm_input)

    # Direct course query
    course_matches = get_course_info(norm_input, course_data)
    if course_matches:
        mem.update_context(norm_input)
        response = "Here are the matching courses:\n\n"
        for match in course_matches:
            response += f"- **{match['course_code']}**: {match['course_title']} ({match['credit_unit']} unit{'s' if match['credit_unit'] != '1' else ''})\n"
        return response

    # Semantic search
    results = semantic_search(norm_input, qa_data, embeddings, model, top_k=1)
    if results:
        top_result = results[0]
        if top_result["score"] > 0.7:
            mem.update_context(norm_input)
            tone = random.choice(["Here’s what I found:", "This might help:", "Check this out:"])
            return f"{tone}\n\n**Q:** {top_result['question']}\n**A:** {top_result['answer']}"

    # GPT fallback
    rewritten = rewrite_question(norm_input, mem.last_context())
    return gpt_fallback(rewritten)

# 🚀 Chat input
user_input = st.chat_input("Ask me anything about Crescent University...")
if user_input:
    add_message("user", user_input)
    bot_typing()
    answer = get_answer(user_input)
    add_message("assistant", answer)
