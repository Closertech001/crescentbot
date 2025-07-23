import streamlit as st
import torch
import re
from sentence_transformers import SentenceTransformer
from utils.embedding import load_dataset, compute_question_embeddings
from utils.course_query import (
    extract_course_query, get_courses_for_query,
    load_course_data, DEPARTMENTS, DEPARTMENT_TO_FACULTY_MAP
)
from utils.greetings import (
    is_greeting, greeting_responses,
    extract_course_code, get_course_by_code,
    is_small_talk, small_talk_response
)
from utils.preprocess import preprocess_text
from utils.tone import get_intro_phrase, get_no_match_phrase
from rapidfuzz import process

# --- Fuzzy department fallback ---
def fuzzy_match_department(text):
    result, score, _ = process.extractOne(text, DEPARTMENTS)
    return result if score >= 80 else None

# --- Deep Follow-up memory handler ---
def update_query_context(follow_up, last_query):
    q = last_query.copy()
    if "second" in follow_up:
        q["semester"] = "Second"
    elif "first" in follow_up:
        q["semester"] = "First"
    if "100" in follow_up:
        q["level"] = "100"
    elif "200" in follow_up:
        q["level"] = "200"
    elif "300" in follow_up:
        q["level"] = "300"
    elif "400" in follow_up:
        q["level"] = "400"
    return q

# --- Load model, QA data, and course data ---
@st.cache_resource
def load_all():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    df = load_dataset("data/crescent_qa.json")
    embeddings = compute_question_embeddings(df["question"].tolist(), model)
    course_data = load_course_data("data/course_data.json")
    return model, df, embeddings, course_data

model, df, embeddings, course_data = load_all()

# --- Semantic search fallback ---
def find_best_match(user_question, model, embeddings, df, threshold=0.6):
    from sentence_transformers.util import cos_sim
    user_embedding = model.encode(user_question.strip().lower(), convert_to_tensor=True)
    cosine_scores = cos_sim(user_embedding, embeddings)[0]
    best_score = torch.max(cosine_scores).item()
    best_idx = torch.argmax(cosine_scores).item()
    if best_score >= threshold:
        return df.iloc[best_idx]["answer"]
    return None

# --- Streamlit UI ---
st.set_page_config(page_title="Crescent University Chatbot", layout="centered")
st.title("🎓 Crescent University Chatbot")
st.markdown("Ask me anything about departments, courses, or university life!")

# --- Session Memory ---
if "chat" not in st.session_state:
    st.session_state.chat = []
if "bot_greeted" not in st.session_state:
    st.session_state.bot_greeted = False
if "last_query_info" not in st.session_state:
    st.session_state.last_query_info = {}

# --- Chat Input ---
USER_AVATAR = "🧑‍💻"
BOT_AVATAR = "🎓"
user_input = st.chat_input("Type your question here...")

if user_input:
    st.session_state.chat.append({"role": "user", "text": user_input})
    cleaned_input = preprocess_text(user_input)
    normalized_input = cleaned_input.lower()

    # --- Greeting ---
    if is_greeting(user_input) and not st.session_state.bot_greeted:
        response = greeting_responses(user_input)
        st.session_state.bot_greeted = True

    # --- Small Talk ---
    elif is_small_talk(user_input):
        response = small_talk_response(user_input)

    # --- Course Code Detection ---
    else:
        course_code = extract_course_code(user_input)
        if course_code:
            course_response = get_course_by_code(course_code, course_data)
            if course_response:
                response = f"📘 *Here’s the info for* `{course_code}`:\n\n{course_response}"
            else:
                response = f"🤔 I couldn't find any details for `{course_code}`. Please check the code and try again."

        else:
            # --- General info / course breakdown ---
            query_info = extract_course_query(normalized_input)

            # Deep Follow-up
            if not any([query_info.get("department"), query_info.get("level"), query_info.get("semester")]):
                last_q = st.session_state.get("last_query_info")
                if last_q:
                    query_info = update_query_context(normalized_input, last_q)

            # Fuzzy department fallback
            if not query_info.get("department"):
                dept_guess = fuzzy_match_department(normalized_input)
                if dept_guess:
                    query_info["department"] = dept_guess.title()
                    query_info["faculty"] = DEPARTMENT_TO_FACULTY_MAP.get(dept_guess)

            # Final decision: structured or semantic
            if query_info and query_info.get("department"):
                result = get_courses_for_query(query_info, course_data)
                st.session_state.last_query_info = query_info
            else:
                result = find_best_match(normalized_input, model, embeddings, df)

            if result:
                response = f"{get_intro_phrase()}\n\n{result}"
            else:
                response = get_no_match_phrase()

    st.session_state.chat.append({"role": "bot", "text": response})

# --- Display chat messages ---
for message in st.session_state.chat:
    avatar = USER_AVATAR if message["role"] == "user" else BOT_AVATAR
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["text"])
