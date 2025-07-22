import streamlit as st
import json
import os
import openai
import faiss
import numpy as np
import re
from datetime import datetime
from sentence_transformers import SentenceTransformer
from textblob import TextBlob
from symspellpy import SymSpell, Verbosity
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# --- Utility Functions ---

def load_dataset(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

def preprocess(text):
    text = text.strip().lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

def compute_question_embeddings(data, model):
    questions = [entry["question"] for entry in data]
    embeddings = model.encode(questions, convert_to_tensor=False)
    return np.array(embeddings)

def build_index(embeddings):
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index

def search(query, model, index, data, top_k=1):
    query_embedding = model.encode([query])
    D, I = index.search(np.array(query_embedding), top_k)
    top_match = data[I[0][0]]
    return top_match, float(D[0][0])

def correct_query(symspell, query):
    suggestions = symspell.lookup(query, Verbosity.CLOSEST, max_edit_distance=2)
    if suggestions:
        return suggestions[0].term
    return query

def get_time_greeting():
    hour = datetime.now().hour
    if hour < 12:
        return "🌅 Good morning"
    elif hour < 18:
        return "🌞 Good afternoon"
    else:
        return "🌙 Good evening"

# Initialize SymSpell
def init_symspell():
    symspell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
    dict_path = "utils/frequency_dictionary_en_82_765.txt"
    symspell.load_dictionary(dict_path, term_index=0, count_index=1)
    return symspell

# GPT fallback
def gpt_fallback(query):
    try:
        res = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": query}]
        )
    except openai.error.OpenAIError:
        res = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": query}]
        )
    return res.choices[0].message.content.strip()

# Emotion and Name detection
def detect_name(text):
    match = re.search(r"\b(?:i[' ]?m|my name is)\s+([A-Z][a-z]+)", text, re.IGNORECASE)
    return match.group(1) if match else None

def detect_emotion(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0.3:
        return "😊 You sound happy!"
    elif polarity < -0.3:
        return "😔 I'm here for you if you need anything."
    return None

# --- Main App ---

def main():
    st.set_page_config(page_title="CrescentBot", page_icon="🎓", layout="centered")
    st.title("🎓 Crescent University Chatbot")
    st.write("Ask me anything about Crescent University!")

    # Load model and dataset
    model = SentenceTransformer("all-MiniLM-L6-v2")
    symspell = init_symspell()

    crescent_data = load_dataset("data/crescent_qa.json")
    course_data = load_dataset("data/course_data.json")
    full_data = crescent_data + course_data

    embeddings = compute_question_embeddings(full_data, model)
    index = build_index(embeddings)

    if "name" not in st.session_state:
        st.session_state.name = None

    user_input = st.chat_input("Type your message...")

    if user_input:
        with st.chat_message("user"):
            st.markdown(user_input)

        corrected_query = correct_query(symspell, preprocess(user_input))
        emotion_note = detect_emotion(user_input)
        name = detect_name(user_input)

        # Name memory
        if name:
            st.session_state.name = name
            with st.chat_message("assistant"):
                st.markdown(f"👋 Nice to meet you, **{name}**!")
            return

        # Handle greetings, thanks, and farewells
        simple_input = preprocess(user_input)
        with st.chat_message("assistant"):
            if any(word in simple_input for word in ["hi", "hello", "hey"]):
                st.markdown(f"{get_time_greeting()}! 👋 How can I assist you today?")
                return
            elif "thank" in simple_input:
                st.markdown("🙏 You're welcome!")
                return
            elif any(word in simple_input for word in ["bye", "goodbye", "see you"]):
                st.markdown("👋 Goodbye! Have a great day.")
                return

        top_match, score = search(corrected_query, model, index, full_data)

        with st.chat_message("assistant"):
            if score < 0.6:
                response = gpt_fallback(user_input)
                st.markdown(f"🤖 *Fallback response:*\n\n{response}")
            else:
                st.markdown(f"📘 *Here’s the info you asked for:*\n\n{top_match['answer']}")

            if emotion_note:
                st.markdown(f"\n\n{emotion_note}")

# Run the app
if __name__ == "__main__":
    main()
