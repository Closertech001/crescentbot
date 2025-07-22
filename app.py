import streamlit as st
import os
import json
import faiss
import numpy as np
import openai
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from textblob import TextBlob

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="🤖 CrescentBot – Crescent University Assistant")

# --- Load JSON datasets safely ---
@st.cache_data
def load_dataset():
    try:
        with open("crescent_qa.json", "r", encoding="utf-8") as f:
            qa_data = json.load(f)
        questions = [entry["question"] for entry in qa_data]
        return qa_data, questions
    except FileNotFoundError:
        st.error("❌ 'crescent_qa.json' not found. Please upload the file to the app directory.")
        return [], []

@st.cache_data
def load_course_data():
    try:
        with open("course_data.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        st.warning("⚠️ 'course_data.json' not found. Some course-specific queries may not work.")
        return {}

# --- Embed questions ---
def compute_question_embeddings(questions, model):
    return model.encode(questions, show_progress_bar=False)

@st.cache_resource
def initialize():
    qa_data, questions = load_dataset()
    course_data = load_course_data()

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embeddings = compute_question_embeddings(questions, model)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))

    return model, index, qa_data, questions, course_data

# --- Semantic search ---
def semantic_search(query, index, model, questions, qa_data, top_k=1):
    query_vec = model.encode([query])
    D, I = index.search(np.array(query_vec), top_k)
    top_idx = I[0][0]
    top_question = questions[top_idx]
    return qa_data[top_idx]["answer"], top_question, float(D[0][0])

# --- OpenAI fallback ---
def fallback_response(query):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful university assistant for Crescent University."},
                {"role": "user", "content": query}
            ],
            temperature=0.4,
        )
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        return "❌ I’m currently unable to fetch a response. Please try again later."

# --- Tone check ---
def detect_sentiment(text):
    return TextBlob(text).sentiment.polarity

# --- MAIN UI ---
st.title("🤖 CrescentBot – Crescent University Assistant")
st.markdown("Ask anything about Crescent University. 📚")

query = st.text_input("Ask me anything:")

if query:
    model, index, qa_data, questions, course_data = initialize()
    sentiment = detect_sentiment(query)
    with st.spinner("Thinking..."):
        if not qa_data:
            st.stop()

        answer, matched_question, score = semantic_search(query, index, model, questions, qa_data)

        if score > 50:  # Higher distance = worse match
            # fallback
            response = fallback_response(query)
            st.markdown(f"💬 *GPT Response:*\n\n{response}")
        else:
            st.markdown(f"💡 *Matched:* `{matched_question}`")
            st.markdown(f"📘 {answer}")
else:
    st.markdown("📝 Tip: Try asking things like *“What are the admission requirements?”* or *“List 200 level Biochemistry courses.”*")

