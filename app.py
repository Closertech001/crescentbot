import streamlit as st
import time
import json
import random
import re
from openai import OpenAI
from sentence_transformers import SentenceTransformer, util
import numpy as np
from rapidfuzz import process, fuzz

# === Load Secrets ===
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# === UI Config ===
st.set_page_config(page_title="Crescent University Bot", layout="wide")
st.markdown("<style> #MainMenu {visibility: hidden;} footer {visibility: hidden;} </style>", unsafe_allow_html=True)

# === Typing animation ===
def typing_animation():
    with st.spinner("Bot is typing..."):
        time.sleep(1.2)

# === Memory tracking ===
if "last_dept" not in st.session_state:
    st.session_state["last_dept"] = None
if "last_level" not in st.session_state:
    st.session_state["last_level"] = None

# === Preprocessing: Replace Pidgin, Slang, Abbreviations, Synonyms ===
from utils.preprocess import normalize_input

# === Load Data ===
with open("data/course_data.json", "r") as f:
    course_data = json.load(f)

with open("data/crescent_qa.json", "r") as f:
    qa_data = json.load(f)

# === Load Embedding Model and Index ===
model = SentenceTransformer("all-MiniLM-L6-v2")
qa_questions = [item["question"] for item in qa_data]
qa_embeddings = model.encode(qa_questions, convert_to_tensor=True)

# === Course Lookup Logic ===
def get_course_by_code(code):
    code = code.replace(" ", "").upper()
    for faculty, depts in course_data.items():
        for dept, levels in depts.items():
            for level, semesters in levels.items():
                for sem, courses in semesters.items():
                    for course in courses:
                        if course["code"].replace(" ", "").upper() == code:
                            return course
    return None

def extract_course_info(dept_name, level=None, semester=None):
    for faculty, depts in course_data.items():
        for dept, levels in depts.items():
            if fuzz.token_sort_ratio(dept_name.lower(), dept.lower()) >= 90:
                if not level:
                    # Return all courses in dept
                    return {
                        "dept": dept,
                        "courses": [
                            c for lvl in levels.values() for sem in lvl.values() for c in sem
                        ]
                    }
                else:
                    level_str = f"{level} level"
                    if level_str in levels:
                        if not semester:
                            return {
                                "dept": dept,
                                "level": level,
                                "courses": [
                                    c for c in levels[level_str]["first"] + levels[level_str]["second"]
                                ]
                            }
                        elif semester in levels[level_str]:
                            return {
                                "dept": dept,
                                "level": level,
                                "semester": semester,
                                "courses": levels[level_str][semester]
                            }
    return None

# === Semantic Search ===
def semantic_search(query, top_k=1):
    query_embedding = model.encode(query, convert_to_tensor=True)
    hits = util.semantic_search(query_embedding, qa_embeddings, top_k=top_k)
    result_idx = hits[0][0]["corpus_id"]
    return qa_data[result_idx]["answer"]

# === Greeting + Small Talk Detection ===
greetings = ["hi", "hello", "hey", "good morning", "good evening"]
farewells = ["bye", "goodbye", "see you", "later"]
small_talk = {
    "how are you": ["I’m fine, thanks!", "Doing great! Ready to assist."],
    "who are you": ["I’m the Crescent University chatbot!", "Your virtual assistant at CUAB."],
    "what can you do": ["I can answer questions about Crescent University, courses, departments, and more."],
}

def handle_small_talk(text):
    for pattern, replies in small_talk.items():
        if pattern in text:
            return random.choice(replies)
    return None

# === Main Logic ===
st.title("🎓 Crescent University Chatbot")

# Input box
user_input = st.text_input("Ask me anything about Crescent University...", key="user_input")

if user_input:
    user_input_norm = normalize_input(user_input)
    response = None

    # Greetings
    if any(greet in user_input_norm for greet in greetings):
        response = random.choice(["Hello there!", "Hi! Ask me anything about CUAB.", "Hey, how can I help?"])

    # Small Talk
    if not response:
        small = handle_small_talk(user_input_norm)
        if small:
            response = small

    # Course Code Detection
    if not response:
        match = re.search(r"\b([A-Z]{2,4}\s?\d{3})\b", user_input.upper())
        if match:
            course_code = match.group(1)
            course = get_course_by_code(course_code)
            if course:
                response = f"**{course['code']} - {course['title']}** ({course['unit']} units)"
            else:
                response = "Hmm, I couldn’t find any course with that code."

    # Follow-up using memory
    if not response:
        dept_match = re.search(r"(?:department of|in)\s+([a-zA-Z ]+)", user_input_norm)
        level_match = re.search(r"(\d{3})\s*level", user_input_norm)
        semester_match = re.search(r"(first|second)\s*semester", user_input_norm)

        dept = dept_match.group(1) if dept_match else st.session_state["last_dept"]
        level = level_match.group(1) if level_match else st.session_state["last_level"]
        semester = semester_match.group(1) if semester_match else None

        if dept:
            st.session_state["last_dept"] = dept
        if level:
            st.session_state["last_level"] = level

        info = extract_course_info(dept, level, semester) if dept else None
        if info:
            response = f"Courses for {info['dept'].title()}"
            if level:
                response += f" - {level} level"
            if semester:
                response += f" ({semester} semester)"
            response += ":\n\n"
            response += "\n".join([f"- {c['code']} - {c['title']} ({c['unit']} units)" for c in info["courses"]])
        else:
            response = None

    # Semantic Search Fallback
    if not response:
        typing_animation()
        try:
            response = semantic_search(user_input_norm)
        except:
            response = "Sorry, I couldn’t understand that. Please try rephrasing."

    # Fallback to GPT
    if not response:
        try:
            typing_animation()
            completion = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "system", "content": "You are a helpful assistant for Crescent University."},
                          {"role": "user", "content": user_input}]
            )
            response = completion.choices[0].message.content
        except Exception as e:
            response = "Sorry, something went wrong with processing your request."

    # Show response
    st.markdown(response)
