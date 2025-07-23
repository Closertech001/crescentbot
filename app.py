import streamlit as st
import time
import random
import re
from utils.preprocess import normalize_input
from utils.course_query import get_course_answer, fuzzy_match_department
from utils.greetings import detect_greeting, get_greeting_response
from utils.embedding import load_qa_data, load_model, compute_embeddings
from utils.search import semantic_search
from utils.memory import MemoryHandler
from openai import OpenAI
import os

# Initialize OpenAI client with your key (set in Streamlit secrets or env var)
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

# Load model and data once (cache for performance)
@st.cache_resource(show_spinner=False)
def load_resources():
    model = load_model()  # SentenceTransformer model, e.g. 'all-MiniLM-L6-v2'
    qa_data = load_qa_data()  # List of dict {question, answer, metadata}
    embeddings = compute_embeddings(model, qa_data)  # Precompute question embeddings
    return model, qa_data, embeddings

model, qa_data, embeddings = load_resources()

# Memory handler for tracking conversation context
memory = MemoryHandler()

# UI Setup
st.set_page_config(page_title="Crescent University Chatbot", page_icon="🎓", layout="wide")

st.title("🎓 Crescent University Chatbot")
st.markdown("Ask me about courses, departments, and admissions at Crescent University!")

# Container for chat messages
chat_container = st.container()

# Input box with default empty string
def get_user_input():
    return st.text_input("Your question here:", key="input_box", placeholder="Type your question and press Enter")

# Typing animation function
def show_typing():
    placeholder = st.empty()
    with placeholder.container():
        for dots in ["Bot is typing.", "Bot is typing..", "Bot is typing..."]:
            st.markdown(f"**{dots}**")
            time.sleep(0.5)
            placeholder.empty()

# Generate dynamic "Here’s what I found for you" style replies
def dynamic_found_intro():
    intros = [
        "Here's what I found for you:",
        "Take a look at this info:",
        "I found this relevant information:",
        "Check this out:",
        "This might help you:"
    ]
    return random.choice(intros)

# GPT fallback call
def call_gpt_fallback(question, chat_history):
    prompt = (
        "You are a helpful assistant specialized in Crescent University courses and admissions.\n"
        "Answer the question concisely and accurately.\n"
        f"Question: {question}\n"
    )
    if chat_history:
        prompt += "Conversation history:\n"
        for user_msg, bot_msg in chat_history:
            prompt += f"User: {user_msg}\nBot: {bot_msg}\n"

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=500,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return "Sorry, I am unable to answer that right now."

# Main chat logic
def handle_user_input(user_input, chat_history):
    user_input_norm = normalize_input(user_input)
    
    # Check for greeting
    if detect_greeting(user_input_norm):
        return get_greeting_response()

    # Try fuzzy department match for department-level queries
    department = fuzzy_match_department(user_input_norm)

    # Try course answer (direct course code or course question)
    course_answer = get_course_answer(user_input_norm)

    # If course answer found, return it immediately
    if course_answer:
        return course_answer

    # If department found, search QA for department info
    if department:
        # Find best semantic match in QA for that department
        matches = semantic_search(user_input_norm, qa_data, embeddings, top_k=3)
        if matches:
            # Pick top match
            best_match = matches[0]
            # Return only the relevant answer, no extra dump
            return f"{dynamic_found_intro()}\n\n{best_match['answer']}"

    # If no course or department info found, fallback to GPT
    fallback_answer = call_gpt_fallback(user_input, chat_history)
    return fallback_answer


def main():
    # Load chat history from session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_input = get_user_input()
    if user_input:
        # Show user message
        with chat_container:
            st.markdown(f"**You:** {user_input}")

        # Show typing animation
        show_typing()

        # Process input and get bot reply
        bot_reply = handle_user_input(user_input, st.session_state.chat_history)

        # Show bot reply
        with chat_container:
            st.markdown(f"**Bot:** {bot_reply}")

        # Append to chat history
        st.session_state.chat_history.append((user_input, bot_reply))

        # Clear input box after sending
        st.session_state["input_box"] = ""

if __name__ == "__main__":
    main()
