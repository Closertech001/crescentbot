import streamlit as st
import os
import json
import faiss
import numpy as np
import openai
from sentence_transformers import SentenceTransformer
from textblob import TextBlob
from dotenv import load_dotenv
from datetime import datetime
import re

# --- Load environment variables ---
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# ---------------- UTILS ---------------- #

def load_dataset(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

def compute_question_embeddings(dataset, model):
    questions = [entry["question"] for entry in dataset]
    embeddings = model.encode(questions, convert_to_tensor=False)
    return np.array(embeddings).astype("float32")

def build_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

def normalize_query(query):
    return re.sub(r'[^\w\s]', '', query.lower())

def correct_typos(query):
    return str(TextBlob(query).correct())

def extract_course_query(text):
    match = re.search(r"\b([A-Z]{3,4}\s?\d{3})\b", text.upper())
    return match.group(1).replace(" ", "") if match else None

def detect_department(text):
    text = text.lower()
    departments = {
        "anatomy": "Department of Anatomy",
        "law": "Department of Law",
        "mass communication": "Department of Mass Communication",
        "computer science": "Department of Computer Science",
        "economics": "Department of Economics and Actuarial Science",
        "accounting": "Department of Accounting",
        "architecture": "Department of Architecture",
        "microbiology": "Department of Microbiology",
        "biochemistry": "Department of Biochemistry",
        "political science": "Department of Political Science & International Studies",
        "physiology": "Department of Physiology"
    }
    for key, val in departments.items():
        if key in text:
            return val
    return None

def search(query, index, model, data, top_k=1):
    query_embedding = model.encode([query], convert_to_tensor=False)
    query_embedding = np.array(query_embedding).astype("float32")
    D, I = index.search(query_embedding, top_k)
    top_idx = I[0][0]
    top_score = D[0][0]
    return data[top_idx], 1 - top_score / (np.linalg.norm(query_embedding) + 1e-5)

def detect_greeting_farewell(query):
    greetings = ["hello", "hi", "good morning", "good afternoon", "good evening", "hey"]
    thanks = ["thank you", "thanks", "thx", "thank u"]
    farewells = ["bye", "goodbye", "see you", "later"]

    norm = query.lower()
    if any(greet in norm for greet in greetings):
        hour = datetime.now().hour
        if hour < 12:
            return "🌅 Good morning! How can I assist you today?"
        elif hour < 17:
            return "🌞 Good afternoon! What would you like to know?"
        else:
            return "🌙 Good evening! Feel free to ask anything about Crescent University."
    elif any(thank in norm for thank in thanks):
        return "😊 You're welcome! Let me know if you need anything else."
    elif any(farewell in norm for farewell in farewells):
        return "👋 Goodbye! Have a great day!"
    return None

def gpt_fallback(query):
    try:
        res = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant for Crescent University. Only answer questions relevant to the university."},
                {"role": "user", "content": query}
            ],
            temperature=0.3
        )
        return res["choices"][0]["message"]["content"].strip()
    except Exception:
        return "❌ Sorry, I'm currently unable to fetch a response from GPT-4."

# ---------------- MAIN APP ---------------- #

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

st.set_page_config(page_title="CrescentBot - University Assistant", layout="wide")
st.markdown(
    "<h2 style='text-align: center; color: #4B8BBE;'>🎓 Crescent University Chatbot</h2>", 
    unsafe_allow_html=True
)

if "history" not in st.session_state:
    st.session_state.history = []

def main():
    data = load_dataset("data/qa_dataset.json")
    model = load_model()
    embeddings = compute_question_embeddings(data, model)
    index = build_index(embeddings)

    st.markdown("##### Ask CrescentBot anything about the university below 👇")
    user_input = st.chat_input("Ask me a question...")

    if user_input:
        st.session_state.history.append({"role": "user", "content": user_input})
        norm_query = normalize_query(correct_typos(user_input))
        quick_response = detect_greeting_farewell(norm_query)

        if quick_response:
            st.session_state.history.append({"role": "assistant", "content": quick_response})
        else:
            top_match, score = search(norm_query, index, model, data, top_k=1)
            course_code = extract_course_query(user_input)
            department = detect_department(norm_query)

            if score > 0.6:
                response = f"📘 *Here’s the info for* `{course_code}`:\n\n{top_match['answer']}" if course_code else top_match["answer"]
            else:
                response = gpt_fallback(norm_query)

            st.session_state.history.append({"role": "assistant", "content": response})

    for chat in st.session_state.history:
        if chat["role"] == "user":
            st.chat_message("user").markdown(f"👤 **You**: {chat['content']}")
        else:
            st.chat_message("assistant").markdown(f"🤖 **CrescentBot**: {chat['content']}")

if __name__ == "__main__":
    main()
