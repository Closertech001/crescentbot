import streamlit as st
import json
import openai
import re

# Custom utility modules
from utils.embedding import load_qa_embeddings, find_most_similar_question
from utils.course_query import extract_course_info
from utils.preprocess import normalize_input
from utils.memory import MemoryHandler
from utils.greetings import detect_greeting, generate_greeting_response, is_small_talk, generate_small_talk_response
from utils.rewrite import rewrite_followup
from utils.search import fallback_to_gpt_if_needed
from utils.tone import detect_tone

# Set up OpenAI API key
try:
    openai.api_key = st.secrets["OPENAI_API_KEY"]
except KeyError:
    st.error("❌ OpenAI API key not found in secrets. Please set OPENAI_API_KEY in .streamlit/secrets.toml")
    st.stop()

# Load data and embeddings
qa_data, question_embeddings = load_qa_embeddings("data/crescent_qa.json")

# Initialize memory
memory = MemoryHandler()

# Confidence threshold for embedding match
CONFIDENCE_THRESHOLD = 0.75


# 🎓 Streamlit app UI
def crescent_chatbot():
    st.set_page_config(page_title="Crescent University Chatbot", layout="centered")
    st.markdown("<h1 style='text-align: center;'>🎓 Crescent University Chatbot 🤖</h1>", unsafe_allow_html=True)

    # Initialize chat session
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display previous messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input from user
    user_input = st.chat_input("Ask me anything about Crescent University...")

    if user_input and user_input.strip():
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = get_bot_response(user_input)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})


# 🧠 Main logic for generating bot reply
def get_bot_response(user_input):
    normalized_input = normalize_input(user_input)

    # Rewrite follow-up using memory
    user_input = rewrite_followup(user_input, memory)

    # Try course-related query first
    course_response = extract_course_info(user_input, memory)
    if course_response:
        return course_response

    # Semantic QA match
    best_match, score = find_most_similar_question(user_input, qa_data, question_embeddings)
    if score > CONFIDENCE_THRESHOLD:
        memory.update_last_topic(best_match)
        return best_match["answer"]

    # Then greetings (only if it's standalone)
    if detect_greeting(normalized_input):
        return generate_greeting_response()

    # Then small talk
    if is_small_talk(normalized_input):
        return generate_small_talk_response()

    # Fallback to GPT
    return fallback_to_gpt_if_needed(user_input)


    # Try semantic QA retrieval
    best_match, score = find_most_similar_question(rewritten_input, qa_data, question_embeddings)
    if score > CONFIDENCE_THRESHOLD:
        memory.update_last_topic(best_match)
        return best_match["answer"]

    # Fallback to GPT for unmatched queries
    return fallback_to_gpt_if_needed(rewritten_input)


# 👇 Launch chatbot
if __name__ == "__main__":
    crescent_chatbot()
