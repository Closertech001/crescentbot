# app.py

import streamlit as st
import time
import torch
from utils.embedding import load_model, load_dataset, compute_question_embeddings
from utils.search import find_response
from utils.rewrite import rewrite_with_tone
from utils.greetings import is_greeting, greeting_responses, extract_course_code, get_course_by_code
from utils.course_query import extract_course_query, get_courses_for_query, load_course_data
from difflib import get_close_matches

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
course_data = load_course_data("data/course_data.json")

# Helper to match similar course codes in case of typos
def fuzzy_match_course_code(input_code, all_course_codes):
    matches = get_close_matches(input_code.upper(), all_course_codes, n=1, cutoff=0.75)
    return matches[0] if matches else None

# Build list of all valid course codes
all_codes = set()
for entry in course_data:
    parts = [part.strip() for part in entry.get("answer", "").split(" | ")]
    for part in parts:
        if ":" in part:
            code = part.split(":")[0].strip()
            all_codes.add(code)

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
        bot_response = greeting_responses(user_query)

    else:
        # Check for course code lookup
        course_code = extract_course_code(user_query)
        if course_code and course_code not in all_codes:
            course_code = fuzzy_match_course_code(course_code, all_codes)

        course_info = get_course_by_code(course_code, course_data) if course_code else None

        if course_info:
            # Try to extract unit count if available
            unit_match = re.search(r"\((\d+) units?\)", course_info)
            unit_text = f" ({unit_match.group(1)} units)" if unit_match else ""
            bot_response = f"**{course_code}** is:
{course_info}{unit_text}"

        else:
            # Check for structured course-related question
            course_query = extract_course_query(user_query)
            matched_courses = get_courses_for_query(course_query, course_data)

            if course_query["department"] and matched_courses:
                heading = f"Here are the courses offered"
                if course_query["level"]:
                    heading += f" in {course_query['level']} level"
                if course_query["semester"]:
                    heading += f" ({course_query['semester']} semester)"
                heading += f" for {course_query['department']}:
"

                bot_response = heading + "\n- " + matched_courses.replace(" | ", "\n- ")

            else:
                # Fallback to semantic search
                with st.spinner("Finding answer..."):
                    response, related_qs = find_response(user_query, model, dataset, embeddings)
                response = rewrite_with_tone(user_query, response)
                bot_response = response

    # Typing animation for assistant response
    with st.chat_message("assistant"):
        placeholder = st.empty()
        animated_response = ""
        for word in bot_response.split():
            animated_response += word + " "
            placeholder.markdown(f'<div class="chat-message-assistant">{animated_response.strip()}</div>', unsafe_allow_html=True)
            time.sleep(0.05)

    st.session_state.chat_history.append({"role": "assistant", "content": bot_response})

    # Show related questions if from embedding path
    if 'related_qs' in locals() and related_qs:
        with st.spinner("Getting related questions..."):
            time.sleep(0.5)
        with st.expander("💡 Related Questions"):
            for q in related_qs:
                st.markdown(f"- {q}")
