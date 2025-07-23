import streamlit as st
import random
import re
import time
from utils.preprocess import normalize_input
from utils.greetings import detect_greeting, detect_small_talk, respond_to_small_talk, GREETING_RESPONSES
from utils.course_query import extract_course_query, get_courses_for_query
from utils.embedding import load_embeddings
from utils.search import find_similar_question
from utils.memory import update_memory, get_last_department_level
from utils.rewrite import rewrite_query_with_memory
from openai import OpenAI

# Load course data and embeddings
course_data, qa_data, qa_embeddings = load_embeddings()

# Initialize OpenAI client
client = OpenAI()

# Streamlit UI setup
st.set_page_config(page_title="Crescent University Chatbot", layout="wide")
st.title("🎓 Crescent University Chatbot")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Greeting tracking
if "greeted" not in st.session_state:
    st.session_state["greeted"] = False

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Typing animation
def show_typing_animation():
    with st.chat_message("assistant"):
        placeholder = st.empty()
        for dots in ["", ".", "..", "..."]:
            placeholder.markdown(f"Bot is typing{dots}")
            time.sleep(0.3)

# Response generation
def generate_response(user_input):
    normalized = normalize_input(user_input)

    # Handle greeting
    if not st.session_state["greeted"] and detect_greeting(normalized):
        st.session_state["greeted"] = True
        return random.choice(GREETING_RESPONSES)

    # Handle small talk
    if detect_small_talk(normalized):
        return respond_to_small_talk(normalized)

    # Try course query
    query_info = extract_course_query(normalized)

    # Add memory if missing
    query_info = rewrite_query_with_memory(query_info)

    # Attempt course lookup
    course_answer = get_courses_for_query(query_info, course_data)
    if course_answer:
        update_memory(query_info)
        return course_answer

    # Semantic fallback
    similar_q, score = find_similar_question(normalized, qa_data, qa_embeddings)
    if score > 0.6:
        return similar_q["answer"]

    # Fallback to GPT
    gpt_response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant for Crescent University students. Respond clearly and concisely."},
            {"role": "user", "content": user_input}
        ]
    )
    return gpt_response.choices[0].message.content.strip()

# Chat input
if prompt := st.chat_input("Ask me anything about Crescent University..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Typing effect
    show_typing_animation()

    response = generate_response(prompt)
    st.chat_message("assistant").markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
