import streamlit as st
import json
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from utils.embedding import load_dataset, compute_question_embeddings
from utils.course_query import extract_course_query, get_courses_for_query, load_course_data
import time
import random
import re

# Load model and dataset
@st.cache_resource
def load_all():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    df = load_dataset("data/crescent_qa.json")
    embeddings = compute_question_embeddings(df["question"].tolist(), model)
    course_data = load_course_data("data/course_data.json")
    return model, df, embeddings, course_data

model, df, embeddings, course_data = load_all()

# Smarter Q&A matcher
def find_best_match(user_question, model, embeddings, df, threshold=0.6, top_k=5):
    from sentence_transformers.util import cos_sim
    user_embedding = model.encode(user_question.strip().lower(), convert_to_tensor=True)
    cosine_scores = cos_sim(user_embedding, embeddings)[0]
    
    top_results = torch.topk(cosine_scores, k=top_k)
    candidates = []
    for idx, score in zip(top_results.indices, top_results.values):
        if score >= threshold:
            row = df.iloc[idx.item()]
            candidates.append((score.item(), row["question"], row["answer"]))
    
    if not candidates:
        return None

    candidates.sort(key=lambda x: (len(x[2]), -x[0]))  # shortest answer, highest score
    return candidates[0][2]

# Greeting recognition
GREETINGS = [
    "hi", "hello", "hey", "good morning", "good afternoon", "good evening"
]
GREET_RESPONSES = [
    "Hello! 👋 How can I help you with Crescent University today?",
    "Hey there! 😊 Ready to answer your questions about CUAB.",
    "Hi! Ask me anything about Crescent University.",
    "Greetings! 👨‍🎓 What would you like to know?",
    "Good to see you! Let’s chat about courses, fees, or anything CUAB."
]
NO_MATCH_RESPONSES = [
    "Hmm, I couldn’t find an answer to that. Could you try rephrasing it? 😊",
    "I'm not sure I got that. Want to try asking it a different way?",
    "That one's a bit tricky. Could you rephrase or be more specific?",
    "Sorry, I don’t have an answer for that yet. Ask me something else!",
    "I didn’t get that one clearly — can you ask it differently?"
]

def is_greeting(text):
    text = text.lower().strip()
    for greet in GREETINGS:
        if re.fullmatch(rf"{greet}[.!]?", text):
            return True
    return False

# Streamlit UI setup
st.set_page_config(page_title="Crescent University Chatbot", layout="centered")
st.title("🎓 Crescent University Chatbot")
st.markdown("Ask me anything about departments, courses, or general university info!")

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

    if is_greeting(normalized_input) and not st.session_state.bot_greeted:
        response = random.choice(GREET_RESPONSES)
        st.session_state.bot_greeted = True
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
            loading_msg = st.empty()
            with loading_msg.container():
                st.markdown("🎓 Bot is typing.")
                time.sleep(0.3)
                st.markdown("🎓 Bot is typing..")
                time.sleep(0.3)
                st.markdown("🎓 Bot is typing...")
                time.sleep(0.4)
            loading_msg.empty()
            response = f"Here’s what I found for you:\n\n{result}"
        else:
            response = random.choice(NO_MATCH_RESPONSES)

    st.session_state.chat.append({"role": "bot", "text": response})

# Display the chat history
for message in st.session_state.chat:
    avatar = USER_AVATAR if message["role"] == "user" else BOT_AVATAR
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["text"])
