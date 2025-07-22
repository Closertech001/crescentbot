# utils/embedding.py

import json
import numpy as np
from sentence_transformers import SentenceTransformer

# Load the model once
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load QA data and compute embeddings
def load_qa_data(filepath="data/crescent_qa.json"):
    with open(filepath, "r", encoding="utf-8") as f:
        qa_pairs = json.load(f)

    questions = [item["question"] for item in qa_pairs]
    embeddings = model.encode(questions, convert_to_tensor=False, normalize_embeddings=True)
    return qa_pairs, np.array(embeddings)

# Get top K matches using cosine similarity
def get_top_k_matches(query, qa_pairs, embeddings, k=3):
    query_embedding = model.encode([query], convert_to_tensor=False, normalize_embeddings=True)[0]
    similarities = np.dot(embeddings, query_embedding)
    top_indices = similarities.argsort()[-k:][::-1]
    top_matches = [qa_pairs[i] for i in top_indices]
    return top_matches
