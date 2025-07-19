import streamlit as st
import json
import time
import random
import os
from textblob import TextBlob
from utils.embedding import load_model, load_dataset, compute_question_embeddings
from utils.search import find_response
from utils.memory import init_memory
from utils.log_utils import log_query
from utils.greetings import is_greeting, greeting_responses

# ----------------------
# App Config
# ----------------------
st.set_page_config(page_title="Crescent University Chatbot", layout="wide")
st.markdown("""
<style>
.chat-message-user {
    background-color: #DCF8C6;
    padding: 10px;
    border-radius: 10px;
    margin-bottom: 10px;
    text-align: right;
}
.chat-message-assistant {
    background-color: #F1F0F0;
    padding: 10px;
    border-radius: 10px;
    margin-bottom: 10px;
    text-align: left;
}
</style>
""", unsafe_allow_html=True)

# ----------------------
# Helper Functions
# ----------------------
def detect_tone(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0.5:
        return "enthusiastic"
    elif 0.1 < polarity <= 0.5:
        return "friendly"
    elif -0.3 <= polarity <= 0.1:
        return "neutral"
    elif -0.6 <= polarity < -0.3:
        return "concerned"
    else:
        return "frustrated"

@st.cache_resource(show_spinner=False)
def get_bot_resources():
    model = load_model()
    dataset = load_dataset("data/crescent_qa.json")
    embeddings = compute_question_embeddings(dataset["question"].tolist(), model)
    return model, dataset, embeddings

# ----------------------
# App Memory Init
# ----------------------
init_memory()
model, dataset, embeddings = get_bot_resources()

# ----------------------
# UI Layout
# ----------------------
st.title("🎓 Crescent University Chatbot")
st.markdown("Ask me anything about Crescent University — courses, departments, units, and more.")
user_input = st.text_input("Type your question here:", key="user_input")

# ----------------------
# Chat Logic
# ----------------------
if user_input:
    with st.spinner("Thinking..."):
        tone = detect_tone(user_input)

        st.session_state.chat_history.append({"role": "user", "content": user_input})
        st.markdown(f'<div class="chat-message-user">{user_input}</div>', unsafe_allow_html=True)

        if is_greeting(user_input):
            response = random.choice(greeting_responses)
        else:
            response, department, score, related = find_response(user_input, dataset, embeddings)
            log_query(user_input, score)
            st.session_state.related_questions = related

            # Tune tone
            if tone == "enthusiastic":
                response = f"🎉 Great question! {response}"
            elif tone == "friendly":
                response = f"Sure! {response}"
            elif tone == "concerned":
                response = f"I understand your concern. Here's what I found: {response}"
            elif tone == "frustrated":
                response = f"I'm really sorry if you're facing issues. Let me help: {response}"

        # Typing animation for main response
        placeholder = st.empty()
        animated_response = ""
        for word in response.split():
            animated_response += word + " "
            placeholder.markdown(f'<div class="chat-message-assistant">{animated_response.strip()}</div>', unsafe_allow_html=True)
            time.sleep(0.05)

        st.session_state.chat_history.append({"role": "assistant", "content": response})

        # Related question loading animation
        if st.session_state.related_questions:
            with st.spinner("Loading related questions..."):
                time.sleep(0.5)
                st.markdown("**Related Questions:**")
                for q in st.session_state.related_questions:
                    st.markdown(f"- {q}")

# ----------------------
# Chat History Display
# ----------------------
if st.session_state.chat_history:
    st.markdown("---")
    st.subheader("Chat History")
    for msg in st.session_state.chat_history:
        role_class = "chat-message-user" if msg["role"] == "user" else "chat-message-assistant"
        st.markdown(f'<div class="{role_class}">{msg["content"]}</div>', unsafe_allow_html=True)
