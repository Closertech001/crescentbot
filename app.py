import streamlit as st
import os
import json
import time
import re
import faiss
import openai
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from textblob import TextBlob
from dotenv import load_dotenv
from datetime import datetime
from symspellpy.symspellpy import SymSpell, Verbosity

# Force CPU usage for compatibility
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# --- Load environment variables ---
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# --- Load QA dataset ---
def load_dataset(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# --- Typo correction ---
def correct_typos(text):
    if not hasattr(correct_typos, "symspell"):
        symspell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
        dict_path = os.path.join(os.path.dirname(__file__), "frequency_dictionary_en_82_765.txt")
        if os.path.exists(dict_path):
            symspell.load_dictionary(dict_path, term_index=0, count_index=1)
        correct_typos.symspell = symspell
    suggestions = correct_typos.symspell.lookup_compound(text, max_edit_distance=2)
    return suggestions[0].term if suggestions else text

# --- Tone detection ---
def detect_tone(user_input):
    polarity = TextBlob(user_input).sentiment.polarity
    if polarity > 0.5:
        return "enthusiastic"
    elif polarity > 0:
        return "friendly"
    elif polarity == 0:
        return "neutral"
    else:
        return "serious"

def respond_in_tone(response, tone):
    if tone == "enthusiastic":
        return response + " 😄"
    elif tone == "friendly":
        return "Sure! " + response
    elif tone == "serious":
        return "Okay. " + response + "..."
    else:
        return response

# --- Semantic Search ---
def compute_embeddings(data, model):
    questions = [entry["question"] for entry in data]
    embeddings = model.encode(questions, convert_to_tensor=False)
    return np.array(embeddings).astype("float32")

def build_faiss_index(embeddings):
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index

def search_semantic(query, model, index, data, top_k=1):
    query_embedding = model.encode([query])[0].astype("float32")
    D, I = index.search(np.array([query_embedding]), top_k)
    if I[0][0] < len(data):
        return data[I[0][0]], D[0][0]
    return None, None

# --- Load embedding model ---
def load_model(name="all-MiniLM-L6-v2"):
    return SentenceTransformer(name)

# --- Load course code ---
def extract_course_query(user_input):
    pattern = r"\b([A-Z]{2,4}\s?\d{3})\b"
    match = re.search(pattern, user_input.upper())
    return match.group(1).replace(" ", "") if match else None

# --- Rewrite to natural ---
def rewrite_input(text):
    replacements = {
        "pls": "please",
        "ur": "your",
        "info": "information",
        "dept": "department",
    }
    return " ".join(replacements.get(word, word) for word in text.split())

# --- Memory store ---
session_memory = []

# --- UI Styling ---
def render_chat(role, text):
    css_class = "user" if role == "user" else "bot"
    st.markdown(
        f'<div class="chat-bubble {css_class}">{text}</div>',
        unsafe_allow_html=True
    )

def show_typing(text, tone):
    placeholder = st.empty()
    final_text = ""
    for char in text:
        final_text += char
        placeholder.markdown(
            f'<div class="chat-bubble bot">{final_text}|</div>',
            unsafe_allow_html=True
        )
        time.sleep(0.02)
    placeholder.markdown(
        f'<div class="chat-bubble bot">{respond_in_tone(text, tone)}</div>',
        unsafe_allow_html=True
    )

# --- GPT fallback ---
def fallback_to_gpt(query):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a university assistant."},
                {"role": "user", "content": query},
            ],
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()
    except:
        return "Sorry, I'm currently unable to fetch a response from GPT-4."

# --- Main ---
def main():
    st.set_page_config("CrescentBot", layout="wide")
    st.markdown(
        """
        <style>
        .chat-bubble {
            max-width: 80%;
            padding: 0.8em 1em;
            margin: 0.5em;
            border-radius: 1em;
            line-height: 1.4;
            font-size: 1rem;
        }
        .user {
            background-color: #DCF8C6;
            align-self: flex-end;
            margin-left: auto;
        }
        .bot {
            background-color: #F1F0F0;
            align-self: flex-start;
            margin-right: auto;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("🎓 Crescent University Assistant")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Load datasets
    qa_data = load_dataset("data/crescent_qa.json")
    course_data = load_dataset("data/course_data.json")

    # Load model & build index
    model = load_model()
    embeddings = compute_embeddings(qa_data, model)
    index = build_faiss_index(embeddings)

    # Chat input
    user_input = st.chat_input("Ask me anything about Crescent University...")

    if user_input:
        corrected = correct_typos(rewrite_input(user_input))
        tone = detect_tone(corrected)
        session_memory.append(corrected)

        render_chat("user", user_input)

        course_code = extract_course_query(corrected)
        if course_code:
            match = next((c for c in course_data if c["code"] == course_code), None)
            if match:
                answer = f"📘 *Here’s the info for* `{course_code}`:\n\n- **Title**: {match['title']}\n- **Level**: {match['level']}\n- **Semester**: {match['semester']}\n- **Department**: {match['department']}"
                show_typing(answer, tone)
            else:
                show_typing("I couldn't find that course in the database.", tone)
        else:
            best_match, score = search_semantic(corrected, model, index, qa_data)
            if best_match and score < 1.0:
                show_typing(best_match["answer"], tone)
            else:
                fallback = fallback_to_gpt(corrected)
                show_typing(fallback, tone)

if __name__ == "__main__":
    main()
