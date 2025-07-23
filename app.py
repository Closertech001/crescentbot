import streamlit as st
import time
import random
import re
from utils.preprocess import normalize_input
from utils.course_query import extract_course_query, get_courses_for_query
from utils.memory import Memory
from utils.embedding import load_model, load_qa_data, compute_embeddings
from utils.search import semantic_search
from utils.greetings import is_greeting, get_greeting_response, is_small_talk, get_small_talk_response
from utils.rewrite import rewrite_with_context
import torch
from openai import OpenAI

# --- Initialize Session State ---
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'memory' not in st.session_state:
    st.session_state.memory = Memory()

# --- Load Embedding Model & Data ---
model = load_model()
df = load_qa_data()
embeddings = compute_embeddings(model, df["question"].tolist())

# --- OpenAI Client ---
openai_client = OpenAI()

# --- Streamlit Page Config ---
st.set_page_config(page_title="Crescent Uni Chatbot", page_icon="🎓")
st.markdown("<h1 style='text-align:center;'>🎓 Crescent University Chatbot</h1>", unsafe_allow_html=True)

# --- Typing animation ---
def simulate_typing(text, delay=0.02):
    message_placeholder = st.empty()
    typed = ""
    for char in text:
        typed += char
        message_placeholder.markdown(f"**🤖 Bot:** {typed}▌")
        time.sleep(delay)
    message_placeholder.markdown(f"**🤖 Bot:** {typed}")

# --- Handle Chat ---
def handle_user_input(user_input):
    # Add user message
    st.session_state.chat_history.append(("user", user_input))

    # Normalize and clean input
    cleaned_input = normalize_input(user_input)

    # Handle greetings
    if is_greeting(cleaned_input):
        bot_response = get_greeting_response()
        st.session_state.chat_history.append(("bot", bot_response))
        simulate_typing(bot_response)
        return

    # Handle small talk
    if is_small_talk(cleaned_input):
        bot_response = get_small_talk_response()
        st.session_state.chat_history.append(("bot", bot_response))
        simulate_typing(bot_response)
        return

    # Extract course info
    dept, level, semester = extract_course_query(cleaned_input)
    if dept:
        st.session_state.memory.update(dept, level, semester)
        courses = get_courses_for_query(dept, level, semester)
        st.session_state.chat_history.append(("bot", courses))
        simulate_typing(courses)
        return

    # Check for course code (e.g., CSC 101)
    code_match = re.search(r"\b[A-Z]{2,4}\s?\d{3}\b", user_input)
    if code_match:
        query = code_match.group(0)
        results = semantic_search(query, df, embeddings, model)
        if results:
            answer = results[0]["answer"]
            st.session_state.chat_history.append(("bot", answer))
            simulate_typing(answer)
            return

    # Else fallback to semantic search + GPT
    results = semantic_search(cleaned_input, df, embeddings, model)
    if results:
        top_result = results[0]
        context = top_result["question"] + "\n" + top_result["answer"]
        rewritten = rewrite_with_context(cleaned_input, context)
        st.session_state.memory.store_last_context(context)

        prompt = f"You are a helpful assistant for Crescent University. Use this context to answer:\n\nContext:\n{context}\n\nQuestion: {rewritten}\n\nAnswer politely and clearly."

        gpt_response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful educational assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5
        )
        final_answer = gpt_response.choices[0].message.content.strip()
        st.session_state.chat_history.append(("bot", final_answer))
        simulate_typing(final_answer)
        return

    # Final fallback
    fallback = "Sorry, I couldn't find anything relevant. Can you rephrase your question?"
    st.session_state.chat_history.append(("bot", fallback))
    simulate_typing(fallback)

# --- Chat Input ---
with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input("Type your message here:", key="user_input")
    submitted = st.form_submit_button("Send")

if submitted and user_input.strip():
    handle_user_input(user_input)

# --- Display Chat History ---
for role, msg in st.session_state.chat_history:
    if role == "user":
        st.markdown(f"**🧑 You:** {msg}")
    else:
        st.markdown(f"**🤖 Bot:** {msg}")

# --- Clear Chat Button ---
st.sidebar.button("🧹 Clear Chat", on_click=lambda: st.session_state.update({"chat_history": [], "memory": Memory()}))
