import streamlit as st
import json
import numpy as np
import openai
import faiss
import os
import re
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from utils.embedding import load_model, load_qa_data, get_question_embeddings, build_faiss_index
from utils.course_query import extract_course_info, get_course_by_code
from utils.memory import update_memory, get_last_context

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Set page config
st.set_page_config(page_title="CrescentBot 🤖", page_icon="🎓", layout="centered")

st.title("🎓 Crescent University Assistant")

# Session state initialization
if "memory" not in st.session_state:
    st.session_state.memory = {}

if "model" not in st.session_state:
    st.session_state.model = load_model("all-MiniLM-L6-v2")

if "qa_data" not in st.session_state:
    st.session_state.qa_data = load_qa_data()

if "questions" not in st.session_state:
    st.session_state.questions = [item["question"] for item in st.session_state.qa_data]

if "embeddings" not in st.session_state:
    st.session_state.embeddings = get_question_embeddings(st.session_state.questions, st.session_state.model)

if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = build_faiss_index(np.array(st.session_state.embeddings))

# Semantic search
def search_similar_question(query, model, index, questions, top_k=1):
    query_embedding = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    scores, indices = index.search(query_embedding, top_k)
    best_match_idx = indices[0][0]
    best_score = scores[0][0]
    return questions[best_match_idx], best_score

# GPT fallback
def fallback_gpt_response(query):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an academic assistant at Crescent University."},
                {"role": "user", "content": query}
            ],
            temperature=0.2
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return "❌ Sorry, I'm currently unable to fetch a response from GPT."

# Handle query
def handle_query(user_input):
    user_input = user_input.strip()
    if not user_input:
        return "Please enter your question."

    # Try course code match first
    course_code_response = get_course_by_code(user_input, course_data)
    if course_code_response:
        return course_code_response

    # Try structured course info query
    structured_response = extract_course_info(user_input, course_data, st.session_state.memory)
    if isinstance(structured_response, str):
        if "Courses for" in structured_response or "No course data" in structured_response:
            return structured_response

    # Try semantic search
    best_question, similarity = search_similar_question(user_input, st.session_state.model, st.session_state.faiss_index, st.session_state.questions)
    if similarity > 0.75:
        for item in st.session_state.qa_data:
            if item["question"] == best_question:
                return item["answer"]

    # Fallback to GPT
    return fallback_gpt_response(user_input)

# Load course data
with open("data/course_data.json", "r", encoding="utf-8") as f:
    course_data = json.load(f)

# Input UI
user_query = st.text_input("💬 Ask CrescentBot a question:")
if st.button("Ask") or user_query:
    response = handle_query(user_query)
    st.markdown(f"**CrescentBot:** {response}")
