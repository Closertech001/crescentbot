import streamlit as st
import json
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from utils.embedding import load_dataset, compute_question_embeddings
from utils.course_query import extract_course_query, get_courses_for_query, load_course_data
from utils.greetings import (
    is_greeting, greeting_responses,
    extract_course_code, get_course_by_code,
    is_small_talk, small_talk_response
)

# Load all cached resources
@st.cache_resource
def load_all():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    df = load_dataset("data/crescent_qa.json")
    embeddings = compute_question_embeddings(df["question"].tolist(), model)
    course_data = load_course_data("data/course_data.json")
    return model, df, embeddings, course_data

model, df, embeddings, course_data = load_all()

# Find closest Q&A match
def find_best_match(user_question, model, embeddings, df, threshold=0.6):
    from sentence_transformers.util import cos_sim
    user_embedding = model.encode(user_question.strip().lower(), convert_to_tensor=True)
    cosine_scores = cos_sim(user_embedding, embeddings)[0]
    best_score = torch.max(cosine_scores).item()
    best_idx = torch.argmax(cosine_scores).item()
    if best_score >= threshold:
        return df.iloc[best_idx]["answer"]
    return None

# Streamlit UI setup
st.set_page_config(page_title="Crescent University Chatbot", layout="centered")
st.title("🎓 Crescent University Chatbot")
st.markdown("Ask me anything about departments, courses, or general university info!")

# Session memory
if "chat" not in st.session_state:
    st.session_state.chat = []
if "bot_greeted" not in st.session_state:
    st.session_state.bot_greeted = False

USER_AVATAR = "🧑‍💻"
BOT_AVATAR = "🎓"

user_input = st.chat_input("Type your question here...")
if user_input:
    st.session_state.chat.append({"role": "user", "text": user_input})
    normalized_input = user_input.lower()

    # Greeting
    if is_greeting(user_input) and not st.session_state.bot_greeted:
        response = greeting_responses(user_input)
        st.session_state.bot_greeted = True

    # Small talk
    elif is_small_talk(user_input):
        response = small_talk_response(user_input)

    # Course code
    else:
        course_code = extract_course_code(user_input)
        if course_code:
            course_response = get_course_by_code(course_code, course_data)
            if course_response:
                response = f"📘 *Here’s the info for* `{course_code}`:\n\n{course_response}"
            else:
                response = f"🤔 I couldn't find any details for `{course_code}`. Please check the code and try again."
        else:
            general_keywords = [
                "admission", "requirement", "fee", "tuition", "duration", "length",
                "cut off", "hostel", "accommodation", "location", "accreditation"
            ]
            if any(word in normalized_input for word in general_keywords):
                result = find_best_match(user_input, model, embeddings, df)
            else:
                query_info = extract_course_query(user_input)
                if query_info and any([query_info.get("department"), query_info.get("level"), query_info.get("semester")]):
                    result = get_courses_for_query(query_info, course_data)
                else:
                    result = find_best_match(user_input, model, embeddings, df)

            if result:
                response = f"✨ Here’s what I found:\n\n{result}"
            else:
                response = "😕 I couldn’t find an answer to that. Try rephrasing it?"

    st.session_state.chat.append({"role": "bot", "text": response})

# Display chat history
for message in st.session_state.chat:
    avatar = USER_AVATAR if message["role"] == "user" else BOT_AVATAR
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["text"])
