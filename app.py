import streamlit as st
import json
import os
import re
import time
import numpy as np
import faiss
import openai
from sentence_transformers import SentenceTransformer
from rapidfuzz import fuzz, process
from datetime import datetime
from textblob import TextBlob
from symspellpy import SymSpell, Verbosity
import emoji

# ----------------- Global Variables -------------------
model = None
index = None
qa_data = None
questions = None
embeddings = None
course_data = None
memory = {"name": None, "department": None, "level": None, "semester": None}

openai.api_key = os.getenv("OPENAI_API_KEY")

# ----------------- Load Model --------------------------
@st.cache_resource(show_spinner=False)
def load_model(name="all-MiniLM-L6-v2"):
    return SentenceTransformer(name)

# ----------------- Load QA dataset ---------------------
def load_qa_data(filepath="data/crescent.json"):
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    questions = [item["question"] for item in data]
    return data, questions

# ----------------- Compute Embeddings ------------------
@st.cache_data(show_spinner=False)
def compute_question_embeddings(questions):
    global model
    return model.encode(questions, convert_to_numpy=True, normalize_embeddings=True)

# ----------------- Build FAISS index -------------------
def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    idx = faiss.IndexFlatIP(dim)  # cosine similarity
    idx.add(embeddings)
    return idx

# ----------------- Semantic Search ---------------------
def semantic_search(query, model, index, qa_data, questions, top_k=1):
    query_vec = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    D, I = index.search(query_vec, top_k)
    idx = I[0][0]
    score = D[0][0]
    return qa_data[idx], score

# ----------------- Course Data & Query -----------------
def load_course_data(filepath="data/course_data.json"):
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

def fuzzy_match_department(dept, course_data, threshold=80):
    all_depts = list(course_data.keys())
    match, score = process.extractOne(dept, all_depts, scorer=fuzz.token_sort_ratio)
    return match if score >= threshold else None

def extract_course_info(user_input):
    global memory, course_data
    dept_match = re.search(r"(?:in|for|of)?\s*(computer science|mass communication|nursing|law|biochemistry|architecture|accounting|microbiology|anatomy|physiology|physics|chemistry)", user_input, re.IGNORECASE)
    level_match = re.search(r"(\d{3}|\d{2,3}-level|\d{3} level|\d{1,3}00|\d{1}-level)", user_input)
    sem_match = re.search(r"(first|1st|second|2nd)\s+semester", user_input.lower())

    department = dept_match.group(1).strip().lower() if dept_match else memory.get("department")
    level = level_match.group(1) if level_match else memory.get("level")
    semester = sem_match.group(1).lower() if sem_match else memory.get("semester")

    if level:
        level = re.findall(r"\d+", level)[0] + "00"
    if semester:
        semester = "first" if "1" in semester or "first" in semester else "second"

    if department:
        department = fuzzy_match_department(department, course_data)
    else:
        return None

    if not department or department not in course_data:
        return "⚠️ I couldn't find that department."

    memory["department"] = department
    if level:
        memory["level"] = level
    if semester:
        memory["semester"] = semester

    dept_data = course_data[department]
    if level not in dept_data:
        return f"📘 No course data found for {level}-level {department.title()}."

    level_data = dept_data[level]

    if semester:
        sem_key = "1st" if "first" in semester else "2nd"
        courses = level_data.get(sem_key, [])
        if not courses:
            return f"📙 No courses found for {semester} semester."
        return f"📘 Courses for {department.title()} {level}-level ({semester} semester):\n" + "\n".join(
            [f"- `{c['code']}`: *{c['title']}* ({c['unit']} units)" for c in courses]
        )
    else:
        courses_1 = level_data.get("1st", [])
        courses_2 = level_data.get("2nd", [])
        response = f"{department.title()} {level}-level courses:\n"
        if courses_1:
            response += "\n📘 First Semester:\n" + "\n".join([f"- `{c['code']}`: *{c['title']}* ({c['unit']} units)" for c in courses_1])
        if courses_2:
            response += "\n\n📗 Second Semester:\n" + "\n".join([f"- `{c['code']}`: *{c['title']}* ({c['unit']} units)" for c in courses_2])
        return response

def get_course_by_code(user_input):
    global course_data
    code_match = re.search(r"\b([A-Z]{2,4}\s?\d{3})\b", user_input.upper())
    if not code_match:
        return None
    course_code = code_match.group(1).replace(" ", "").upper()
    for dept, levels in course_data.items():
        for level, semesters in levels.items():
            for sem, courses in semesters.items():
                for course in courses:
                    if course["code"].replace(" ", "").upper() == course_code:
                        return f"📘 `{course_code}` is *{course['title']}* offered in `{dept.title()}`, {level}-level ({sem} semester), worth {course['unit']} unit(s)."
    return "⚠️ I couldn't find any course matching that code."

# ----------------- Greetings & Tone --------------------
def get_time_based_greeting():
    hour = datetime.now().hour
    if hour < 12:
        return "Good morning 🌞"
    elif hour < 18:
        return "Good afternoon ☀️"
    else:
        return "Good evening 🌙"

def detect_tone(text):
    # Simplified tone detection
    text_lower = text.lower()
    if any(word in text_lower for word in ["thank", "thanks"]):
        return "gratitude"
    elif any(word in text_lower for word in ["bye", "see you", "goodnight"]):
        return "farewell"
    elif any(word in text_lower for word in ["hello", "hi", "hey"]):
        return "greeting"
    return "neutral"

def respond_to_tone(tone):
    if tone == "gratitude":
        return "You're welcome! 😊"
    elif tone == "farewell":
        return "Goodbye! Take care! 👋"
    elif tone == "greeting":
        return get_time_based_greeting() + " How can I assist you today?"
    else:
        return None

# ----------------- Typing animation --------------------
def typewriter_effect(text, delay=0.03):
    placeholder = st.empty()
    full_text = ""
    for char in text:
        full_text += char
        placeholder.markdown(full_text + "▌")
        time.sleep(delay)
    placeholder.markdown(full_text)

# ----------------- Main App -----------------------------
def main():
    st.set_page_config(page_title="CrescentBot - Crescent University Assistant", page_icon="🤖")
    st.markdown(
        """
        <style>
        .user-msg {
            background-color: #dcf8c6;
            border-radius: 10px;
            padding: 8px 15px;
            margin: 5px 0;
            max-width: 80%;
            align-self: flex-end;
        }
        .bot-msg {
            background-color: #f1f0f0;
            border-radius: 10px;
            padding: 8px 15px;
            margin: 5px 0;
            max-width: 80%;
            align-self: flex-start;
        }
        .chat-container {
            display: flex;
            flex-direction: column;
            max-width: 700px;
            margin: 0 auto;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        </style>
        """, unsafe_allow_html=True
    )

    global model, index, qa_data, questions, embeddings, course_data

    # Load model & datasets once
    if model is None:
        model = load_model()
    if qa_data is None:
        qa_data, questions = load_qa_data()
    if embeddings is None:
        embeddings = compute_question_embeddings(questions)
    if index is None:
        index = build_faiss_index(embeddings)
    if course_data is None:
        course_data = load_course_data()

    st.title("🤖 CrescentBot - Crescent University Assistant")

    # Session state chat history
    if "history" not in st.session_state:
        st.session_state.history = []

    # User input
    query = st.text_input("Ask me anything about Crescent University:", key="input", placeholder="Type your question here...")

    if query:
        # Display user message
        st.session_state.history.append({"sender": "user", "message": query})

        # Check tone-based quick replies (greetings, thanks, bye)
        tone = detect_tone(query)
        tone_response = respond_to_tone(tone)

        if tone_response:
            response = tone_response
        else:
            # Check if it's a course code query
            course_resp = get_course_by_code(query)
            if course_resp:
                response = course_resp
            else:
                # Check for course info query
                course_info_resp = extract_course_info(query)
                if course_info_resp and isinstance(course_info_resp, str) and not course_info_resp.startswith("⚠️"):
                    response = course_info_resp
                else:
                    # Semantic search fallback
                    best_match, score = semantic_search(query, model, index, qa_data, questions)
                    if score > 0.6:
                        response = best_match["answer"]
                    else:
                        # GPT fallback - simplified for demo
                        response = "Sorry, I don't have an answer for that yet. Please ask something else."

        # Add bot response
        st.session_state.history.append({"sender": "bot", "message": response})

        # Clear input
        st.session_state.input = ""

    # Display chat history
    container = st.container()
    with container:
        for chat in st.session_state.history:
            if chat["sender"] == "user":
                st.markdown(f"<div class='user-msg'>You: {emoji.emojize(chat['message'], use_aliases=True)}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='bot-msg'>CrescentBot: {emoji.emojize(chat['message'], use_aliases=True)}</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
