import streamlit as st
import json
import torch
import numpy as np
import re
from sentence_transformers import SentenceTransformer
from utils.embedding import load_dataset, compute_question_embeddings
from utils.course_query import extract_course_query, get_courses_for_query, load_course_data, DEPARTMENTS, DEPARTMENT_TO_FACULTY_MAP
from utils.greetings import (
    is_greeting,
    get_greeting_response,
    is_small_talk,
    get_small_talk_response
)
from openai import OpenAI

client = OpenAI()

# Load Q&A data and embeddings
qa_data, qa_embeddings = load_qa_embeddings()
memory = MemoryHandler()

# Typing animation
def typing_animation():
    with st.spinner("Bot is typing..."):
        time.sleep(1)

# Initialize Streamlit page
st.set_page_config(page_title="Crescent University Chatbot", page_icon="🎓")
st.markdown("<h2 style='text-align: center;'>Crescent University Chatbot 🤖</h2>", unsafe_allow_html=True)

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Input box
user_input = st.chat_input("Ask me anything about Crescent University...")

if user_input:
    normalized_input = normalize_input(user_input)
    st.session_state.chat_history.append(("user", user_input))

    # --- GREETING OR SMALL TALK ---
    if is_greeting(normalized_input):
        response = get_greeting_response()

    elif is_small_talk(normalized_input):
        response = get_small_talk_response(normalized_input)

    else:
        # --- COURSE OR DEPARTMENT QUERY ---
        course_answer = handle_course_query(normalized_input, memory)
        if course_answer:
            response = course_answer

        else:
            # --- SEMANTIC SEARCH ---
            match = semantic_search(normalized_input, qa_data, qa_embeddings)
            if match:
                response = match
            else:
                # --- FALLBACK TO GPT ---
                typing_animation()
                rewritten = rewrite_question(normalized_input)
                gpt_reply = client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant for Crescent University."},
                        {"role": "user", "content": rewritten}
                    ],
                    temperature=0.7
                )
                response = gpt_reply.choices[0].message.content

    st.session_state.chat_history.append(("bot", response))

# Display chat
for role, msg in st.session_state.chat_history:
    if role == "user":
        st.chat_message("user").markdown(msg)
    else:
        st.chat_message("assistant").markdown(msg)
