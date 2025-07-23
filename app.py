import streamlit as st
import json
import re
import torch
import pkg_resources
import pandas as pd
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from symspellpy import SymSpell, Verbosity
from rapidfuzz import process

# ----------------------------
# 🔠 Abbreviation + Synonym Maps
# ----------------------------
ABBREVIATIONS = {
    "u": "you", "r": "are", "ur": "your", "cn": "can", "cud": "could",
    "shud": "should", "wud": "would", "abt": "about", "bcz": "because",
    "plz": "please", "pls": "please", "tmrw": "tomorrow", "wat": "what",
    "wats": "what is", "info": "information", "yr": "year", "sem": "semester",
    "admsn": "admission", "clg": "college", "sch": "school", "uni": "university",
    "cresnt": "crescent", "l": "level", "d": "the", "msg": "message",
    "dept": "department", "pg": "postgraduate", "app": "application", "req": "requirement",
    "1st": "first", "2nd": "second", "fee": "fees", "csc": "computer science",
    "mass comm": "mass communication", "acc": "accounting"
}

SYNONYMS = {
    "hod": "head of department", "school": "university", "college": "faculty",
    "class": "course", "subject": "course", "unit": "credit", "hostel": "accommodation",
    "lodging": "accommodation", "room": "accommodation", "school fees": "tuition",
    "acceptance fee": "admission fee", "fees": "tuition", "enrol": "apply",
    "join": "apply", "sign up": "apply", "admit": "apply", "requirement": "criteria"
}

# ----------------------------
# 🔄 Preprocessing
# ----------------------------
@st.cache_resource
def get_sym_spell():
    sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
    dict_path = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")
    sym_spell.load_dictionary(dict_path, term_index=0, count_index=1)
    return sym_spell

def preprocess_text(text):
    text = re.sub(r'[^\w\s\-]', '', text)
    text = re.sub(r'(.)\1{2,}', r'\1', text)
    words = text.lower().split()
    words = [ABBREVIATIONS.get(w, w) for w in words]
    sym_spell = get_sym_spell()
    corrected = [sym_spell.lookup(w, Verbosity.CLOSEST, max_edit_distance=2)[0].term if sym_spell.lookup(w, Verbosity.CLOSEST, max_edit_distance=2) else w for w in words]
    final = [SYNONYMS.get(w, w) for w in corrected]
    return ' '.join(final)

# ----------------------------
# 📁 Data Loading
# ----------------------------
def load_dataset(path="data/crescent_qa.json"):
    with open(path, "r", encoding="utf-8") as f:
        return pd.DataFrame(json.load(f))

def load_course_data(path="data/course_data.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def compute_question_embeddings(questions, model):
    return model.encode(questions, convert_to_tensor=True, show_progress_bar=False)

# ----------------------------
# 🎯 Course Code Lookup
# ----------------------------
def extract_course_code(text):
    match = re.search(r"[A-Z]{2,4}[-]?[A-Z]{0,3}\s?\d{3}", text.upper())
    return match.group(0).replace(" ", "") if match else None

def get_course_by_code(code, course_data):
    for item in course_data:
        if code.lower() in item.get("question", "").lower():
            return item["answer"]
    return None

# ----------------------------
# 💬 Greeting / Small Talk
# ----------------------------
GREETINGS = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"]
def is_greeting(text): return any(g in text.lower() for g in GREETINGS)
def greeting_responses(_): return "🎓 Hello! What would you like to know about Crescent University?"

def is_small_talk(text):
    patterns = [r"how (are|r) (you|u)", r"what'?s up", r"are you (okay|fine)"]
    return any(re.search(p, text.lower()) for p in patterns)
def small_talk_response(_): return "I'm doing great! 😊 Ask me anything about the university."

# ----------------------------
# 🏛 Department Fuzzy Logic
# ----------------------------
DEPARTMENTS = [
    "computer science", "anatomy", "biochemistry", "accounting",
    "business administration", "political science and international studies",
    "microbiology", "economics with operations research", "mass communication",
    "law", "nursing", "physiology", "architecture"
]

DEPARTMENT_TO_FACULTY_MAP = {
    "computer science": "CONAS", "anatomy": "COHES", "biochemistry": "CONAS",
    "accounting": "CASMAS", "business administration": "CASMAS",
    "political science and international studies": "CASMAS", "microbiology": "CONAS",
    "economics with operations research": "CASMAS", "mass communication": "CASMAS",
    "law": "BACOLAW", "nursing": "COHES", "physiology": "COHES", "architecture": "COES"
}

def fuzzy_match_department(text):
    result, score, _ = process.extractOne(text, DEPARTMENTS)
    return result if score >= 80 else None

def extract_course_query(text):
    text = preprocess_text(text)
    level = re.search(r"\b(100|200|300|400)\s*level\b", text)
    semester = re.search(r"\b(first|second)\s*semester\b", text)
    dept = fuzzy_match_department(text)
    return {
        "level": level.group(1) if level else None,
        "semester": semester.group(1).capitalize() if semester else None,
        "department": dept.title() if dept else None,
        "faculty": DEPARTMENT_TO_FACULTY_MAP.get(dept) if dept else None
    }

def get_courses_for_query(query, course_data):
    if not query: return None
    dept = query.get("department", "").lower()
    level = query.get("level", "").lower() if query.get("level") else None
    semester = query.get("semester", "").lower() if query.get("semester") else None
    matches = []
    for item in course_data:
        if item.get("department", "").lower() != dept:
            continue
        if level and item.get("level", "").lower() != level:
            continue
        if semester and semester not in item.get("question", "").lower():
            continue
        matches.append(item)
    if not matches: return None
    return matches[0]["answer"]

# ----------------------------
# 🧠 Deep Follow-up Context
# ----------------------------
def update_query_context(follow_up, last_query):
    q = last_query.copy()
    if "second" in follow_up: q["semester"] = "Second"
    elif "first" in follow_up: q["semester"] = "First"
    if "100" in follow_up: q["level"] = "100"
    elif "200" in follow_up: q["level"] = "200"
    elif "300" in follow_up: q["level"] = "300"
    elif "400" in follow_up: q["level"] = "400"
    return q

def semantic_search(question, model, embeddings, df, threshold=0.6):
    user_embedding = model.encode(question, convert_to_tensor=True)
    cosine_scores = cos_sim(user_embedding, embeddings)[0]
    best_score = torch.max(cosine_scores).item()
    best_idx = torch.argmax(cosine_scores).item()
    if best_score >= threshold:
        return df.iloc[best_idx]["answer"]
    return None

def random_intro():
    intros = ["Here’s what I found for you 😊:", "Let’s break it down 🔍:", "Sure! Here's the info 📘:"]
    import random
    return random.choice(intros)

# ----------------------------
# 🌐 Streamlit Chat App Starts Here
# ----------------------------
@st.cache_resource
def load_all():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    df = load_dataset()
    embeddings = compute_question_embeddings(df["question"].tolist(), model)
    course_data = load_course_data()
    return model, df, embeddings, course_data

model, df, embeddings, course_data = load_all()

st.set_page_config(page_title="Crescent University Chatbot", layout="centered")
st.title("🎓 Crescent University Chatbot")
st.markdown("Ask me anything about departments, courses, or general university info!")

if "chat" not in st.session_state: st.session_state.chat = []
if "bot_greeted" not in st.session_state: st.session_state.bot_greeted = False
if "last_query_info" not in st.session_state: st.session_state.last_query_info = {}

USER_AVATAR = "🧑‍💻"
BOT_AVATAR = "🎓"

user_input = st.chat_input("Ask a question...")
if user_input:
    st.session_state.chat.append({"role": "user", "text": user_input})
    normalized_input = preprocess_text(user_input)

    if is_greeting(user_input) and not st.session_state.bot_greeted:
        response = greeting_responses(user_input)
        st.session_state.bot_greeted = True
    elif is_small_talk(user_input):
        response = small_talk_response(user_input)
    else:
        course_code = extract_course_code(user_input)
        if course_code:
            course_response = get_course_by_code(course_code, course_data)
            response = f"📘 *Here’s the info for* `{course_code}`:\n\n{course_response}" if course_response else f"🤔 I couldn't find any details for `{course_code}`."
        else:
            query_info = extract_course_query(user_input)
            if not any([query_info.get("department"), query_info.get("level"), query_info.get("semester")]):
                last_q = st.session_state.last_query_info
                if last_q:
                    query_info = update_query_context(normalized_input, last_q)
            if query_info.get("department"):
                response = get_courses_for_query(query_info, course_data)
                st.session_state.last_query_info = query_info
            else:
                response = semantic_search(normalized_input, model, embeddings, df)
            response = f"{random_intro()}\n\n{response}" if response else "😕 I couldn’t find an answer to that. Try rephrasing it?"

    st.session_state.chat.append({"role": "bot", "text": response})

for msg in st.session_state.chat:
    avatar = USER_AVATAR if msg["role"] == "user" else BOT_AVATAR
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["text"])
