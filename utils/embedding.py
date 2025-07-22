# utils/embedding.py

import json
import numpy as np
from sentence_transformers import SentenceTransformer, util

# Load sentence transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

def load_qa_embeddings(qa_path="data/crescent_qa.json"):
    """Load questions and compute their embeddings."""
    with open(qa_path, "r", encoding="utf-8") as f:
        qa_data = json.load(f)

    questions = [item["question"] for item in qa_data]
    embeddings = model.encode(questions, convert_to_tensor=True)
    return qa_data, embeddings

def find_most_similar_question(user_input, qa_data, embeddings, top_k=3):
    """Return top-k most similar questions and their scores."""
    query_embedding = model.encode(user_input, convert_to_tensor=True)
    hits = util.semantic_search(query_embedding, embeddings, top_k=top_k)[0]

    results = []
    for hit in hits:
        index = hit["corpus_id"]
        score = hit["score"]
        results.append((qa_data[index], score))
    return results
