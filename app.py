import streamlit as st
from utils.course_query import extract_course_query, load_course_data, get_courses_for_query
import random
import time
import re

# Load course data once
course_data = load_course_data("data/course_data.json")

# Greetings and farewells
GREETINGS = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"]
GREET_RESPONSES = [
    "Hi there! 👋", "Hello! How can I help you today?", "Hey! 😊 What would you like to know?"
]
FAREWELLS = ["bye", "goodbye", "see you", "take care", "later"]
FAREWELL_RESPONSES = [
    "Goodbye! 👋", "Take care!", "See you later!", "Bye! Have a great day!"
]

# Typing simulation
def bot_typing(delay=1.5):
    with st.empty():
        st.markdown("**Bot is typing...**")
        time.sleep(delay)

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    st.session_state.started = False

# App UI setup
st.set_page_config(page_title="Crescent Chatbot", layout="centered")
st.title("🎓 Crescent University Course Chatbot")

with st.sidebar:
    st.markdown("### 🤖 Chat Info")
    st.markdown("This bot can help you find **courses** by department, level, and semester.")
    if st.button("🧹 Clear Chat"):
        st.session_state.chat_history = []
        st.session_state.started = False
        st.experimental_rerun()

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Greet user on first open
if not st.session_state.started:
    greeting = random.choice(GREET_RESPONSES)
    st.chat_message("assistant").markdown(greeting)
    st.session_state.chat_history.append({"role": "assistant", "content": greeting})
    st.session_state.started = True

# Input box for user prompt
user_input = st.chat_input("Ask about courses (e.g. 'What are 200 level Nursing courses?')")

# Handle user message
if user_input:
    st.chat_message("user").markdown(user_input)
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    lower_input = user_input.strip().lower()

    # Check for farewell
    if any(f in lower_input for f in FAREWELLS):
        farewell = random.choice(FAREWELL_RESPONSES)
        bot_typing()
        st.chat_message("assistant").markdown(farewell)
        st.session_state.chat_history.append({"role": "assistant", "content": farewell})

    # Check for greeting with no intent
    elif any(greet in lower_input for greet in GREETINGS) and len(lower_input.split()) <= 3:
        response = random.choice(GREET_RESPONSES)
        bot_typing()
        st.chat_message("assistant").markdown(response)
        st.session_state.chat_history.append({"role": "assistant", "content": response})

    else:
        # Handle actual query
        query_info = extract_course_query(user_input)
        result = get_courses_for_query(query_info, course_data)

        bot_typing()

        if result:
            st.chat_message("assistant").markdown(result)
            st.session_state.chat_history.append({"role": "assistant", "content": result})
        else:
            fallback = "🤔 I couldn’t find any matching course info. Please check the department, level, or semester."
            st.chat_message("assistant").markdown(fallback)
            st.session_state.chat_history.append({"role": "assistant", "content": fallback})
