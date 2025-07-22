import streamlit as st
import json
import os
import openai
import time
from sentence_transformers import SentenceTransformer
from utils.embedding import load_dataset, get_question_embeddings, build_faiss_index
from utils.course_query import extract_course_query
from utils.semantic_search import search_semantic
from utils.tone import detect_tone
from utils.greetings import check_greeting
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="CrescentBot 🤖", layout="wide")
st.markdown("""
    <style>
    .user-msg {
        background-color: #DCF8C6;
        border-radius: 20px;
        padding: 10px 15px;
        margin: 10px;
        max-width: 70%;
        align-self: flex-end;
    }
    .bot-msg {
        background-color: #F1F0F0;
        border-radius: 20px;
        padding: 10px 15px;
        margin: 10px;
        max-width: 70%;
        align-self: flex-start;
    }
    .chat-box {
        display: flex;
        flex-direction: column;
        height: 70vh;
        overflow-y: auto;
        padding: 10px;
        border: 1px solid #ccc;
        border-radius: 10px;
        background-color: #fff;
    }
    </style>
""", unsafe_allow_html=True)

# Preload model and data
@st.cache_resource
def initialize():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    qa_data, questions = load_dataset("data/crescent_qa.json")
    embeddings = get_question_embeddings(questions, model)
    index = build_faiss_index(embeddings)

    with open("data/course_data.json", "r", encoding="utf-8") as f:
        course_data = json.load(f)

    return model, index, qa_data, questions, course_data

model, index, qa_data, questions, course_data = initialize()

st.title("🤖 CrescentBot — Your Campus Assistant")
st.markdown("""Ask me anything about Crescent University! 🏫✨""")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

prompt = st.chat_input("Type your question here...")

if prompt:
    # Detect greeting
    if check_greeting(prompt):
        response = "👋 Hello! I'm CrescentBot. How can I assist you today?"
    else:
        # Typing effect
        with st.spinner("Thinking..."):
            course_code, course_response = extract_course_query(prompt, course_data)
            if course_response:
                response = course_response
            else:
                result = search_semantic(prompt, model, index, qa_data, questions)
                if result:
                    response = result
                else:
                    try:
                        gpt_reply = openai.ChatCompletion.create(
                            model="gpt-4",
                            messages=[
                                {"role": "system", "content": "You are CrescentBot, a helpful assistant for students of Crescent University."},
                                {"role": "user", "content": prompt}
                            ]
                        )
                        response = gpt_reply["choices"][0]["message"]["content"]
                    except:
                        response = "❌ Sorry, I'm currently unable to fetch a response."

        tone = detect_tone(prompt)
        tone_personality = {
            "friendly": "😊 Hey there! CrescentBot here to help. ",
            "formal": "Good day. Here's the information you requested: ",
            "casual": "Yo! Here’s the scoop: ",
            "funny": "😄 Let me crack the code for you real quick: ",
            "neutral": ""
        }
        prefix = tone_personality.get(tone, "")
        response = prefix + response

    # Save to chat history
    st.session_state.chat_history.append((prompt, response))

# Display chat messages
st.markdown('<div class="chat-box">', unsafe_allow_html=True)
for user_msg, bot_msg in st.session_state.chat_history:
    st.markdown(f'<div class="user-msg">🧑‍💻 {user_msg}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="bot-msg">🤖 {bot_msg}</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)
