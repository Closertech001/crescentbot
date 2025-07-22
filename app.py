import streamlit as st
import json
import os
import re
import time
import random
from datetime import datetime
from collections import deque

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from rapidfuzz import fuzz, process
from symspellpy import SymSpell, Verbosity
import openai
import emoji

from dotenv import load_dotenv
load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

# ==== Utils Inlined ==== #

# --- Embedding utils ---
@st.cache_resource(show_spinner=False)
def load_model(model_name="all-MiniLM-L6-v2"):
    return SentenceTransformer(model_name)

@st.cache_data(show_spinner=False)
def load_dataset(qa_path="data/crescent_qa.json", course_path="data/course_data.json"):
    with open(qa_path, "r", encoding="utf-8") as f:
        qa_data = json.load(f)
    with open(course_path, "r", encoding="utf-8") as f:
        course_data = json.load(f)
    questions = [entry["question"] for entry in qa_data]
    return qa_data, questions, course_data

@st.cache_data(show_spinner=False)
def compute_question_embeddings(questions, model):
    embeddings = model.encode(questions, convert_to_numpy=True, normalize_embeddings=True)
    return embeddings

@st.cache_resource(show_spinner=False)
def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine similarity
    index.add(embeddings)
    return index

def semantic_search(query_embedding, index, qa_data, top_k=1):
    D, I = index.search(query_embedding, top_k)
    best_idx = I[0][0]
    score = D[0][0]
    return qa_data[best_idx], score

# --- SymSpell typo correction ---
@st.cache_resource(show_spinner=False)
def load_symspell():
    max_edit_distance_dictionary = 2
    prefix_length = 7
    sym_spell = SymSpell(max_edit_distance_dictionary, prefix_length)
    dict_path = os.path.join(os.path.dirname(__file__), "frequency_dictionary_en_82_765.txt")
    if os.path.exists(dict_path):
        sym_spell.load_dictionary(dict_path, term_index=0, count_index=1)
    else:
        st.warning("SymSpell dictionary file frequency_dictionary_en_82_765.txt not found.")
    return sym_spell

def correct_text(sym_spell, text):
    suggestions = sym_spell.lookup_compound(text, max_edit_distance=2)
    if suggestions:
        return suggestions[0].term
    return text

# --- Course query utils ---

department_faculty_map = {
    "computer science": "CICOT",
    "information technology": "CICOT",
    "mass communication": "CASMAS",
    "accounting": "CASMAS",
    "business administration": "CASMAS",
    "architecture": "COES",
    "nursing": "COHES",
    "physiology": "COHES",
    "anatomy": "COHES",
    "physics": "CONAS",
    "chemistry": "CONAS",
    "microbiology": "CONAS",
    "biochemistry": "CONAS",
    "law": "BACOLAW"
}

def fuzzy_match_department(dept, course_data, threshold=80):
    all_depts = list(course_data.keys())
    match, score = process.extractOne(dept, all_depts, scorer=fuzz.token_sort_ratio)
    return match if score >= threshold else None

def extract_course_info(user_input, course_data, memory):
    dept_match = re.search(r"(?:in|for|of)?\s*(\b[a-zA-Z ]{3,}\b department)?\s*(computer science|mass communication|nursing|law|biochemistry|architecture|accounting|microbiology|anatomy|physiology|physics|chemistry)", user_input, re.IGNORECASE)
    level_match = re.search(r"(\d{3}|\d{2,3}-level|\d{3} level|\d{1,3}00|\d{1}-level)", user_input)
    sem_match = re.search(r"(first|1st|second|2nd)\s+semester", user_input.lower())

    department = None
    if dept_match and dept_match.group(2):
        department = dept_match.group(2).strip().lower()
    elif dept_match:
        department = dept_match.group(3).lower()
    else:
        department = memory.get("department")

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
    if level: memory["level"] = level
    if semester: memory["semester"] = semester

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

def get_course_by_code(user_input, course_data):
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

# --- Tone detection and personality ---

def detect_tone(text):
    text_lower = text.lower()
    if any(w in text_lower for w in ["thank", "thanks", "thx"]):
        return "gratitude"
    elif any(w in text_lower for w in ["bye", "goodbye", "see you", "later"]):
        return "farewell"
    elif any(w in text_lower for w in ["hello", "hi", "hey", "greetings"]):
        return "greeting"
    # Simple frustration detection
    elif any(w in text_lower for w in ["stupid", "bad", "hate", "angry", "frustrat", "annoy"]):
        return "frustrated"
    return "neutral"

def personality_response(tone, name=None):
    if tone == "gratitude":
        return "You're very welcome! 😊"
    elif tone == "farewell":
        return "Bye! Have a great day! 👋"
    elif tone == "greeting":
        greet = "Hello"
        if name:
            greet += f", {name}"
        greet += "! How can I help you today? 🤖"
        return greet
    elif tone == "frustrated":
        return "I'm sorry you're feeling frustrated. Let me try to help! 🙏"
    else:
        return None

# --- GPT Fallback ---

def gpt_fallback(prompt, personality=None):
    messages = []
    system_msg = "You are CrescentBot, a helpful university assistant. Keep responses concise and polite."
    if personality == "frustrated":
        system_msg += " The user is frustrated, be extra polite and helpful."
    messages.append({"role": "system", "content": system_msg})
    messages.append({"role": "user", "content": prompt})
    try:
        completion = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages,
            temperature=0.7,
            max_tokens=300,
            n=1,
            stop=None,
        )
        answer = completion.choices[0].message.content.strip()
        return answer
    except Exception as e:
        # fallback to GPT-3.5-turbo
        try:
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=0.7,
                max_tokens=300,
                n=1,
                stop=None,
            )
            answer = completion.choices[0].message.content.strip()
            return answer
        except Exception as e:
            return "Sorry, I'm having trouble reaching the AI service right now."

# --- Memory (simple dict in session) ---

def update_memory(memory, key, value):
    memory[key] = value

def get_memory(memory, key, default=None):
    return memory.get(key, default)

# ==== Streamlit UI ==== #

st.set_page_config(page_title="CrescentBot – Crescent University Assistant", page_icon="🤖", layout="wide")
st.markdown(
    """
    <style>
    /* Chat container */
    .chat-container {
        max-width: 800px;
        margin: auto;
        padding: 1rem;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .user-message {
        background: #0084ff;
        color: white;
        padding: 10px 15px;
        border-radius: 18px 18px 0 18px;
        max-width: 70%;
        margin-left: auto;
        margin-bottom: 10px;
        font-size: 16px;
        white-space: pre-wrap;
    }
    .bot-message {
        background: #e5e5ea;
        color: black;
        padding: 10px 15px;
        border-radius: 18px 18px 18px 0;
        max-width: 70%;
        margin-right: auto;
        margin-bottom: 10px;
        font-size: 16px;
        white-space: pre-wrap;
    }
    .typing {
        font-style: italic;
        color: #666666;
        padding-left: 10px;
    }
    .emoji {
        font-size: 18px;
    }
    </style>
    """, unsafe_allow_html=True
)

# Load everything once
model = load_model()
qa_data, questions, course_data = load_dataset()
embeddings = compute_question_embeddings(questions, model)
index = build_faiss_index(embeddings)
sym_spell = load_symspell()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = deque(maxlen=30)
if "memory" not in st.session_state:
    st.session_state.memory = {}
if "name" not in st.session_state:
    st.session_state.name = None

def simulate_typing(text, delay=0.02):
    # Yields text char by char to simulate typing animation
    typed = ""
    for c in text:
        typed += c
        yield typed
        time.sleep(delay)

def main():
    st.title("🤖 CrescentBot – Crescent University Assistant")
    st.markdown("Ask me anything about Crescent University courses, departments, policies, and more!")

    # Display chat history
    chat_container = st.container()

    with chat_container:
        for i, chat in enumerate(st.session_state.chat_history):
            is_user = chat["role"] == "user"
            msg_class = "user-message" if is_user else "bot-message"
            st.markdown(f'<div class="{msg_class}">{emoji.emojize(chat["content"])}</div>', unsafe_allow_html=True)

    query = st.text_input("Your question:", key="input_text", placeholder="Type your question here...", label_visibility="collapsed")

    if st.button("Send") or (query and st.session_state.get("input_text_send", False)):
        user_input = st.session_state.input_text.strip()
        if not user_input:
            st.warning("Please enter a question.")
            return

        # Clear input box
        st.session_state.input_text = ""

        # SymSpell correction
        corrected_input = correct_text(sym_spell, user_input)

        # Check for greetings, farewells, gratitude first (simple responses)
        tone = detect_tone(corrected_input)
        name = st.session_state.get("name")

        if tone == "greeting" and not any(m["role"] == "bot" and "Hello" in m["content"] for m in st.session_state.chat_history):
            bot_resp = personality_response(tone, name)
        elif tone in ["gratitude", "farewell"]:
            bot_resp = personality_response(tone, name)
        else:
            # Try course code exact match first
            course_resp = get_course_by_code(corrected_input, course_data)
            if course_resp:
                bot_resp = course_resp
            else:
                # Try course info extraction
                course_info_resp = extract_course_info(corrected_input, course_data, st.session_state.memory)
                if course_info_resp and not course_info_resp.startswith("⚠️"):
                    bot_resp = course_info_resp
                else:
                    # Semantic search + threshold
                    query_embedding = model.encode([corrected_input], convert_to_numpy=True, normalize_embeddings=True)
                    best_match, score = semantic_search(query_embedding, index, qa_data)
                    if score > 0.6:
                        bot_resp = best_match["answer"]
                    else:
                        bot_resp = gpt_fallback(corrected_input, tone)

        # Save chat history
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        # Typing animation effect for bot
        bot_msg_placeholder = st.empty()

        for partial in simulate_typing(bot_resp, delay=0.015):
            bot_msg_placeholder.markdown(f'<div class="bot-message">{emoji.emojize(partial)}</div>', unsafe_allow_html=True)

        st.session_state.chat_history.append({"role": "bot", "content": bot_resp})

if __name__ == "__main__":
    main()
