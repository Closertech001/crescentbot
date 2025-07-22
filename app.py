import streamlit as st
import json
import os
import re
import time
import numpy as np
import faiss
import openai
from sentence_transformers import SentenceTransformer
from symspellpy import SymSpell, Verbosity
from textblob import TextBlob
from dotenv import load_dotenv

# --- Load environment variables ---
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# --- Custom CSS for polished UI ---
CUSTOM_CSS = """
<style>
body {
    background-color: #f7f9fc;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}
div[data-testid="stMarkdownContainer"] {
    font-size: 16px;
    line-height: 1.5;
}
.stChatMessage[data-testid="stChatMessage"][data-role="user"] div[data-testid="stMarkdownContainer"] {
    background-color: #4a90e2;
    color: white;
    padding: 12px 16px;
    border-radius: 20px 20px 0 20px;
    max-width: 75%;
    margin-left: auto;
    margin-bottom: 8px;
    white-space: pre-wrap;
}
.stChatMessage[data-testid="stChatMessage"][data-role="assistant"] div[data-testid="stMarkdownContainer"] {
    background-color: #e2e8f0;
    color: #1a202c;
    padding: 12px 16px;
    border-radius: 20px 20px 20px 0;
    max-width: 75%;
    margin-right: auto;
    margin-bottom: 8px;
    white-space: pre-wrap;
}
div[role="textbox"] {
    font-size: 16px !important;
    padding: 12px !important;
}
div[data-testid="stScrollableContainer"]::-webkit-scrollbar {
    width: 8px;
}
div[data-testid="stScrollableContainer"]::-webkit-scrollbar-thumb {
    background-color: #a0aec0;
    border-radius: 20px;
}
.stChatMessage div[data-testid="stMarkdownContainer"] {
    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
}
.emojione {
    font-size: 1.2em !important;
}
</style>
"""

# --- SymSpell Setup for typo correction ---
max_edit_distance_dictionary = 2
prefix_length = 7
sym_spell = SymSpell(max_edit_distance_dictionary, prefix_length)
dictionary_path = "frequency_dictionary_en_82_765.txt"
if os.path.exists(dictionary_path):
    sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
else:
    st.warning("⚠️ SymSpell dictionary not found. Spell correction disabled.")

# --- Load Crescent QA dataset ---
def load_dataset(filepath="data/crescent.json"):
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    questions = [entry["question"] for entry in data]
    return data, questions

# --- Load course data ---
def load_course_data(filepath="data/course_data.json"):
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

# --- Load model ---
@st.cache_resource(show_spinner=False)
def load_model(name="all-MiniLM-L6-v2"):
    return SentenceTransformer(name)

# --- Compute embeddings ---
@st.cache_data(show_spinner=False)
def compute_question_embeddings(questions, model):
    return model.encode(questions, convert_to_numpy=True, normalize_embeddings=True)

# --- Build FAISS index ---
@st.cache_resource(show_spinner=False)
def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # Cosine similarity with normalized embeddings
    index.add(embeddings)
    return index

# --- SymSpell correction ---
def correct_query(query):
    if not sym_spell:
        return query
    suggestions = sym_spell.lookup_compound(query, max_edit_distance=2)
    if suggestions:
        return suggestions[0].term
    return query

# --- Simple greeting detection ---
def detect_greeting(text):
    greetings = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"]
    return any(g in text.lower() for g in greetings)

# --- Gratitude detection ---
def detect_gratitude(text):
    thanks_words = ["thank you", "thanks", "thx", "ty"]
    return any(t in text.lower() for t in thanks_words)

# --- Farewell detection ---
def detect_farewell(text):
    farewells = ["bye", "goodbye", "see you", "farewell"]
    return any(f in text.lower() for f in farewells)

# --- Tone detection (simple placeholder) ---
def detect_tone(text):
    # Just a simple heuristic for demonstration
    lower = text.lower()
    if any(w in lower for w in ["please", "kindly"]):
        return "formal"
    if any(w in lower for w in ["lol", "haha", "funny"]):
        return "humorous"
    return "neutral"

# --- Semantic search ---
def semantic_search(query, index, questions, embeddings, top_k=1):
    query_emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    D, I = index.search(query_emb, top_k)
    idx = I[0][0]
    score = D[0][0]
    return questions[idx], score

# --- GPT fallback ---
def gpt_fallback(query, tone="neutral"):
    style_prompt = {
        "formal": "Respond politely and formally.",
        "humorous": "Respond with humor and lightheartedness.",
        "neutral": "Respond clearly and helpfully."
    }
    prompt = style_prompt.get(tone, "Respond clearly and helpfully.")
    messages = [
        {"role": "system", "content": f"You are a helpful assistant. {prompt}"},
        {"role": "user", "content": query}
    ]
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages,
            temperature=0.7,
            max_tokens=300,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        # fallback to GPT-3.5
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=0.7,
                max_tokens=300,
            )
            return response.choices[0].message.content.strip()
        except Exception as e2:
            return "Sorry, I’m having trouble answering right now."

# --- Main app ---
def main():
    st.set_page_config(page_title="🤖 CrescentBot – Crescent University Assistant", page_icon="🤖")
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    st.title("🤖 CrescentBot – Crescent University Assistant")

    # Load datasets and model (cached)
    data, questions = load_dataset()
    course_data = load_course_data()
    global model
    model = load_model()
    embeddings = compute_question_embeddings(questions, model)
    index = build_faiss_index(embeddings)

    if "history" not in st.session_state:
        st.session_state.history = []
    if "memory" not in st.session_state:
        st.session_state.memory = {}

    # Chat input
    user_input = st.text_input("Ask me anything about Crescent University:", key="input", on_change=None)

    if user_input:
        # Clear input box after submit
        st.session_state.input = ""

        # Correct spelling
        corrected_query = correct_query(user_input)

        # Handle greetings, thanks, farewells
        if detect_greeting(corrected_query):
            bot_response = "Hello! How can I assist you today? 😊"
        elif detect_gratitude(corrected_query):
            bot_response = "You're welcome! Glad I could help. 🙌"
        elif detect_farewell(corrected_query):
            bot_response = "Goodbye! Have a great day! 👋"
        else:
            tone = detect_tone(corrected_query)
            # Semantic search
            top_question, score = semantic_search(corrected_query, index, questions, embeddings)
            if score > 0.6:
                # Find answer from dataset
                answer = next((item["answer"] for item in data if item["question"] == top_question), None)
                bot_response = answer
            else:
                # GPT fallback
                bot_response = gpt_fallback(corrected_query, tone)

        # Append messages to chat history
        st.session_state.history.append({"role": "user", "content": user_input})
        st.session_state.history.append({"role": "assistant", "content": bot_response})

    # Display chat history
    for chat in st.session_state.history:
        role = chat["role"]
        content = chat["content"]
        if role == "user":
            st.markdown(f"**You:** {content}  ")
        else:
            st.markdown(f"**CrescentBot:** {content}  ")

if __name__ == "__main__":
    main()
