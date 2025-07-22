import streamlit as st
import os
import json
import faiss
import numpy as np
import openai
from sentence_transformers import SentenceTransformer
from textblob import TextBlob
from symspellpy import SymSpell
from dotenv import load_dotenv
from datetime import datetime
import re
import time

# --- Load environment variables ---
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# --- Initialize model & data ---
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load datasets
def load_dataset(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

# Combine both datasets
data = load_dataset("data/crescent.json") + load_dataset("data/course_data.json")

# --- Preprocessing ---
questions = [entry["question"] for entry in data]
answers = [entry["answer"] for entry in data]

# Build FAISS index
embeddings = model.encode(questions)
index = faiss.IndexFlatL2(len(embeddings[0]))
index.add(np.array(embeddings))

# --- SymSpell Setup ---
symspell = SymSpell(max_dictionary_edit_distance=2)
symspell.load_dictionary("frequency_dictionary_en_82_765.txt", term_index=0, count_index=1)

def correct_spelling(text):
    suggestions = symspell.lookup_compound(text, max_edit_distance=2)
    return suggestions[0].term if suggestions else text

# --- GPT fallback ---
def gpt_fallback(query):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are Crescent University's helpful assistant."},
                {"role": "user", "content": query}
            ],
            temperature=0.4,
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return "⚠️ Sorry, I couldn’t fetch a response right now."

# --- Semantic Search ---
def semantic_search(query, top_k=1):
    query_embedding = model.encode([query])
    scores, indices = index.search(np.array(query_embedding), top_k)
    top_index = indices[0][0]
    return data[top_index], 1 - scores[0][0] if len(scores[0]) > 0 else 0

# --- Tone, Greeting, Name ---
def detect_tone_and_greeting(message):
    lower = message.lower()
    hour = datetime.now().hour

    if any(word in lower for word in ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"]):
        if hour < 12:
            return "🌅 Good morning!"
        elif hour < 17:
            return "☀️ Good afternoon!"
        else:
            return "🌆 Good evening!"

    if "thank" in lower:
        return "🙏 You’re very welcome!"

    if any(word in lower for word in ["bye", "see you", "take care"]):
        return "👋 Bye for now!"

    name_match = re.search(r"\b(i[' ]?m|my name is|call me)\s+([A-Z][a-z]+)", message, re.I)
    if name_match:
        st.session_state["user_name"] = name_match.group(2).strip()
        return f"😊 Nice to meet you, {st.session_state['user_name']}!"

    blob = TextBlob(message)
    polarity = blob.sentiment.polarity
    if polarity < -0.3:
        return "😟 I’m really sorry to hear that. Want to talk about it?"
    elif polarity > 0.4:
        return "😄 That’s great to hear!"

    return None

# --- App UI ---
st.set_page_config(page_title="CrescentBot 🎓", page_icon="🤖")
st.title("🤖 CrescentBot – Ask Me Anything!")

if "user_name" not in st.session_state:
    st.session_state.user_name = None

def main():
    with st.chat_message("assistant"):
        st.markdown("Hi there! I’m **CrescentBot**. How can I help you today? 😊")

    user_query = st.chat_input("Ask me about Crescent University...")

    if user_query:
        st.chat_message("user").markdown(user_query)

        with st.spinner("CrescentBot is typing..."):
            time.sleep(1.4)

            # Correct spelling
            corrected = correct_spelling(user_query)
            norm_query = corrected.strip()

            # Greeting & Tone
            tone_reply = detect_tone_and_greeting(norm_query)
            if tone_reply:
                st.chat_message("assistant").markdown(tone_reply)
                return

            # Semantic search
            match, score = semantic_search(norm_query)
            if score > 0.6:
                response = match["answer"]
            else:
                response = gpt_fallback(norm_query)

            # Personalize with name
            if st.session_state.user_name:
                response = f"👤 *{st.session_state.user_name}*, {response}"

            st.chat_message("assistant").markdown(response)

if __name__ == "__main__":
    main()
