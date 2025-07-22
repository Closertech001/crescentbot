# utils/embedding.py

import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import streamlit as st

@st.cache_data(show_spinner="🔍 Loading model and embedding index...")
def load_model_and_embeddings(json_path="data/crescent_qa.json"):
    model = SentenceTransformer("all-MiniLM-L6-v2")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    questions = [item["question"].strip().lower() for item in data]
    embeddings = model.encode(questions, convert_to_numpy=True)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    return model, data, index, embeddings

def find_most_similar_question(user_input, model, index, qa_data):
    query_vector = model.encode([user_input], convert_to_numpy=True)
    D, I = index.search(query_vector, k=1)

    best_index = I[0][0]
    best_score = D[0][0]
    best_match = qa_data[best_index]

    # Normalize distance score to similarity (optional)
    similarity = 1 / (1 + best_score) if best_score != 0 else 1.0

    return best_match, similarity

