# utils/semantic_search.py

import numpy as np

def search_semantic(query, model, index, questions, top_k=1):
    """
    Perform semantic search using FAISS and return the best matching question and score.
    """
    # Encode query into embedding
    query_embedding = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)

    # Perform search
    D, I = index.search(query_embedding, top_k)

    # Handle case where no result is found
    if I[0][0] == -1:
        return None, 0.0

    # Return top match and score
    top_match = questions[I[0][0]]
    score = float(D[0][0])
    return top_match, score
