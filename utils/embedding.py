import json
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ✅ Cache model to avoid reloading every time
@st.cache_resource(show_spinner="🔍 Loading embedding model...")
def load_model(model_name="all-MiniLM-L6-v2"):
    return SentenceTransformer(model_name)

# ✅ Load model globally
model = load_model()

# ✅ Load QA data and compute question embeddings
@st.cache_data(show_spinner="🔎 Embedding QA pairs...")
def load_qa_data(filepath="data/crescent_qa.json"):
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    questions = [item["question"] for item in data]
    embeddings = model.encode(questions, show_progress_bar=False)
    return data, np.array(embeddings)

# ✅ Find top K similar matches using cosine similarity
def get_top_k_matches(user_query, qa_data, qa_embeddings, k=3):
    query_embedding = model.encode([user_query])
    similarities = cosine_similarity(query_embedding, qa_embeddings)[0]
    top_k_indices = similarities.argsort()[-k:][::-1]
    results = []

    for idx in top_k_indices:
        results.append({
            "question": qa_data[idx]["question"],
            "answer": qa_data[idx]["answer"],
            "score": float(similarities[idx])
        })

    return results
