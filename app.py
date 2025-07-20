import streamlit as st
import json
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from utils.embedding import load_dataset, compute_question_embeddings
from utils.course_query import extract_course_query, get_courses_for_query, load_course_data

# Load model and dataset
@st.cache_resource
def load_all():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    df = load_dataset("data/crescent_qa.json")
    embeddings = compute_question_embeddings(df["question"].tolist(), model)
    course_data = load_course_data("data/course_data.json")
    return model, df, embeddings, course_data

model, df, embeddings, course_data = load_all()

# Function to find best matching question
def find_best_match(user_question, model, embeddings, df, threshold=0.6):
    from sentence_transformers.util import cos_sim
    user_embedding = model.encode(user_question.strip().lower(), convert_to_tensor=True)
    cosine_scores = cos_sim(user_embedding, embeddings)[0]
    best_score = torch.max(cosine_scores).item()
    best_idx = torch.argmax(cosine_scores).item()
    if best_score >= threshold:
        return df.iloc[best_idx]["answer"]
    return None

# UI setup
st.set_page_config(page_title="Crescent University Chatbot", layout="centered")
st.title("🎓 Crescent University Chatbot")
st.markdown("Ask me anything about departments, courses, or general university info!")

# Session state
if "chat" not in st.session_state:
    st.session_state.chat = []
if "bot_greeted" not in st.session_state:
    st.session_state.bot_greeted = False

# Handle chat input
user_input = st.chat_input("Type your question here...")
if user_input:
    st.session_state.chat.append({"role": "user", "text": user_input})

    # Greeting logic
    greetings = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"]
    if any(greet in user_input.lower() for greet in greetings) and not st.session_state.bot_greeted:
        response = "Hello! 👋 How can I help you with Crescent University today?"
        st.session_state.bot_greeted = True

    else:
        # Check course-related query
        query_info = extract_course_query(user_input)
        if query_info and any([query_info.get("department"), query_info.get("level"), query_info.get("semester")]):
            result = get_courses_for_query(query_info, course_data)
        else:
            result = None

        # Fallback to general Q&A
        if not result:
            result = find_best_match(user_input, model, embeddings, df)

        # Still nothing found
        if not result:
            result = "I'm sorry, I couldn't find an answer to that. Try rephrasing your question."

        response = result

    st.session_state.chat.append({"role": "bot", "text": response})

# Display chat history with avatars
for message in st.session_state.chat:
    if message["role"] == "user":
        with st.chat_message("user", avatar="🧑‍🎓"):  # or use a custom image URL
            st.markdown(message["text"])
    else:
        with st.chat_message("assistant", avatar="🤖"):  # or a logo image
            st.markdown(message["text"])
