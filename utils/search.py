# utils/search.py

import numpy as np
import faiss

def semantic_search_faiss(query, model, index, qa_data, questions, top_k=3):
    query_vec = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    scores, indices = index.search(query_vec, top_k)
    matches = []
    for i in range(top_k):
        idx = indices[0][i]
        if idx < len(qa_data):
            matches.append({
                "question": questions[idx],
                "answer": qa_data[idx]["answer"],
                "score": float(scores[0][i]),
                "department": qa_data[idx].get("department", "")
            })
    return matches
