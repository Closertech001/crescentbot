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

def find_most_similar_question(user_query, model, index, qa_data, top_k=1):
    """
    Perform FAISS-based vector search to find the most similar question.

    Args:
        user_query (str): User input.
        model: SentenceTransformer model.
        index: FAISS index.
        qa_data: List of QA pairs.
        top_k (int): Number of top results to return.

    Returns:
        (dict, float): Best matched QA pair and similarity score.
    """
    query_vec = model.encode([user_query], convert_to_numpy=True)
    distances, indices = index.search(query_vec, top_k)

    if indices[0][0] == -1:
        return None, 0.0  # No match found

    matched_qa = qa_data[indices[0][0]]
    similarity_score = 1 / (1 + distances[0][0])  # Convert L2 distance to similarity

    return matched_qa, similarity_score
