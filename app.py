# app.py

import streamlit as st
import time
import torch
from utils.embedding import load_model, load_dataset, compute_question_embeddings
from utils.search import find_response
from utils.rewrite import rewrite_with_tone
from utils.greetings import is_greeting, greeting_responses

# App configuration
st.set_page_config(page_title="Crescent University Chatbot", layout="wide")
st.title("🎓 Crescent University Chatbot")

# Session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

@st.cache_resource(show_spinner=False)
def get_bot_resources():
    model = load_model()
    dataset = load_dataset("data/crescent_qa.json")
    embeddings = compute_question_embeddings(dataset["question"].tolist(), model)
    return model, dataset, embeddings

model, dataset, embeddings = get_bot_resources()

# User Input
user_query = st.chat_input("Type your question here:")

# Chat Output Display
for chat in st.session_state.chat_history:
    with st.chat_message(chat["role"]):
        st.markdown(chat["content"])

# Process Input
if user_query:
    with st.chat_message("user"):
        st.markdown(user_query)
    st.session_state.chat_history.append({"role": "user", "content": user_query})

    # Check if greeting
    if is_greeting(user_query):
        bot_response = greeting_responses()
    else:
        # Search answer
        with st.spinner("Finding answer..."):
            response, related_qs = find_response(user_query, model, dataset, embeddings)

        # Apply tone rewriting if needed
        response = rewrite_with_tone(user_query, response)

        # Typing animation for assistant response
        with st.chat_message("assistant"):
            placeholder = st.empty()
            animated_response = ""
            for word in response.split():
                animated_response += word + " "
                placeholder.markdown(f'<div class="chat-message-assistant">{animated_response.strip()}</div>', unsafe_allow_html=True)
                time.sleep(0.05)

        st.session_state.chat_history.append({"role": "assistant", "content": response})

        # Loading animation for related questions
        if related_qs:
            with st.spinner("Getting related questions..."):
                time.sleep(0.5)
            with st.expander("💡 Related Questions"):
                for q in related_qs:
                    st.markdown(f"- {q}")
