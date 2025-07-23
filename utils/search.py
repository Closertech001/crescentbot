# utils/search.py

import openai
import os

# utils/search.py

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def find_similar_question(query, model, embeddings, qa_data, threshold=0.75, top_k=1):
    query_embedding = model.encode([query], convert_to_numpy=True)
    similarities = cosine_similarity(query_embedding, embeddings)[0]

    top_indices = similarities.argsort()[::-1][:top_k]
    results = []

    for idx in top_indices:
        if similarities[idx] >= threshold:
            results.append({
                "question": qa_data[idx]["question"],
                "answer": qa_data[idx]["answer"],
                "score": float(similarities[idx])
            })

    return results
