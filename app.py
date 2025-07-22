# app.py

import streamlit as st
import json
import random
import re
import datetime
from sentence_transformers import SentenceTransformer
import numpy as np

# ========== Setup ==========

st.set_page_config(page_title="Crescent University Chatbot", layout="centered")

# Typing animation
typing_placeholder = st.empty()

# ========== Greeting Detection ==========

def detect_greeting(text):
    greetings = ["hello", "hi", "hey", "good morning", "good afternoon", "good evening"]
    return any(greet in text.lower() for greet in greetings)

def get_random_greeting():
    return random.choice(["Hello!", "Hi there!", "Greetings!", "Hey! 👋"])

def detect_farewell(text):
    farewells = ["bye", "goodbye", "see you", "later", "farewell"]
    return any(farewell in text.lower() for farewell in farewells)

# ========== Tone Detection & Rewrite ==========

def detect_tone(text):
    text = text.lower()
    if any(word in text for word in ["pls", "please", "hi", "hello", "thank"]):
        return "polite"
    if any(word in text for word in ["urgent", "now", "quick", "fast", "immediately", "asap"]):
        return "urgent"
    if any(word in text for word in ["why", "what", "how", "confused", "help", "explain", "not sure"]):
        return "confused"
    if any(word in text for word in ["angry", "nonsense", "rubbish", "annoyed", "frustrated"]):
        return "angry"
    if re.search(r"[!?]{2,}", text):
        return "emphatic"
    return "neutral"

def rewrite_with_tone(user_input, response):
    tone = detect_tone(user_input)
    if tone == "polite":
        return "Sure! 😊 " + response
    elif tone == "urgent":
        return "Got it — here's the information you need right away:\n\n" + response
    elif tone == "confused":
        return "No worries, let me explain clearly:\n\n" + response
    elif tone == "angry":
        return "I'm here to help — let's sort this out calmly:\n\n" + response
    elif tone == "emphatic":
        return "Absolutely! Here's everything you need:\n\n" + response
    else:
        return "Here's what I found for you:\n\n" + response

# ========== Memory ==========

memory = []

def update_memory(query):
    memory.append({
        "timestamp": datetime.datetime.now().isoformat(),
        "query": query
    })

def get_last_context():
    return memory[-1]["query"] if memory else ""

# ========== Load Course Data ==========

with open("data/course_data.json", "r", encoding="utf-8") as f:
    course_data = json.load(f)

department_mapping = {
    "computer science": "CICOT",
    "mass communication": "CASMAS",
    "nursing": "COHES",
    "law": "BACOLAW",
    "architecture": "COES",
    "physics": "CONAS"
    # Add more as needed
}

# ========== Normalize Input ==========

def normalize_input(text):
    text = text.lower().strip()
    replacements = {
        "comp sci": "computer science",
        "mass comm": "mass communication",
        "nurs": "nursing",
        "phys": "physics",
        "archi": "architecture"
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text

# ========== Course Query ==========

def get_course_info(user_input, course_data, dept_map):
    user_input = user_input.lower()
    for dept, faculty in dept_map.items():
        if dept in user_input:
            level_match = re.search(r"\b(100|200|300|400|500)\b", user_input)
            semester_match = re.search(r"\b(first|second)\b", user_input)

            level = level_match.group(1) if level_match else None
            semester = semester_match.group(1) if semester_match else None

            dept_data = course_data.get(faculty, {}).get(dept.title(), {})

            if level and semester:
                return format_course_response(dept_data.get(level, {}).get(semester, {}), dept, level, semester)
            elif level:
                return format_level_response(dept_data.get(level, {}), dept, level)
            else:
                return f"What level or semester are you asking about for {dept.title()}?"
    return None

def format_course_response(courses, dept, level, semester):
    if not courses:
        return f"No courses found for {dept.title()} {level} level {semester} semester."
    result = f"{dept.title()} {level} Level - {semester.title()} Semester Courses:\n"
    for course in courses:
        result += f"- {course['code']} - {course['title']} ({course['unit']} units)\n"
    return result

def format_level_response(semesters, dept, level):
    if not semesters:
        return f"No course data available for {dept.title()} {level} level."
    result = f"{dept.title()} {level} Level Courses:\n"
    for sem, courses in semesters.items():
        result += f"\n{sem.title()} Semester:\n"
        for course in courses:
            result += f"- {course['code']} - {course['title']} ({course['unit']} units)\n"
    return result

# ========== Embedding + Similarity Search ==========

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_data
def load_qa_data():
    with open("data/crescent_qa.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    questions = [q["question"] for q in data]
    answers = [q["answer"] for q in data]
    return questions, answers

def get_top_k_matches(query, questions, answers, model, k=3):
    embeddings = model.encode(questions)
    query_embedding = model.encode([query])[0]
    scores = np.dot(embeddings, query_embedding) / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_embedding)
    )
    top_k_idx = np.argsort(scores)[::-1][:k]
    return [{"question": questions[i], "answer": answers[i], "score": float(scores[i])} for i in top_k_idx]

model = load_model()
questions, answers = load_qa_data()

# ========== Handle Input ==========

def handle_input(user_input):
    user_input_norm = normalize_input(user_input)

    if detect_greeting(user_input_norm):
        return get_random_greeting()
    if detect_farewell(user_input_norm):
        return "Goodbye! Have a great day."

    course_response = get_course_info(user_input_norm, course_data, department_mapping)
    if course_response:
        return rewrite_with_tone(user_input, course_response)

    last_context = get_last_context()
    full_input = f"{last_context} {user_input_norm}" if last_context else user_input_norm

    update_memory(user_input_norm)

    matches = get_top_k_matches(full_input, questions, answers, model)
    if matches and matches[0]["score"] > 0.6:
        return rewrite_with_tone(user_input, matches[0]["answer"])
    else:
        return rewrite_with_tone(user_input, "Sorry, I couldn't find an answer for that.")

# ========== UI ==========

st.title("🎓 Crescent University Chatbot")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).markdown(msg["content"])

user_input = st.chat_input("Ask me anything about Crescent University...")

if user_input:
    st.session_state["messages"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    typing_placeholder.markdown("🤖 *Bot is typing...*")
    response = handle_input(user_input)
    typing_placeholder.empty()

    st.session_state["messages"].append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)
