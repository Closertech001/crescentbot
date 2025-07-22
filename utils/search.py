# utils/search.py

import numpy as np

def cosine_similarity(a, b):
    """
    Compute the cosine similarity between two sets of vectors.
    a: (n, dim), b: (1, dim) or (dim,)
    """
    if b.ndim == 1:
        b = b.reshape(1, -1)
    return np.dot(a, b.T).squeeze()

def semantic_search(user_query, qa_data, embeddings, model, top_k=3):
    """
    Perform semantic search to find top_k most relevant Q&A entries.
    
    - user_query: str
    - qa_data: list of Q&A dicts
    - embeddings: numpy array of question embeddings
    - model: SentenceTransformer
    - top_k: number of results to return

    Returns:
        List of top_k matching Q&A dictionaries
    """
    query_embedding = model.encode(
        user_query, convert_to_numpy=True, normalize_embeddings=True
    )

    scores = cosine_similarity(embeddings, query_embedding)
    top_indices = np.argsort(scores)[-top_k:][::-1]

    top_results = []
    for idx in top_indices:
        match = qa_data[idx]
        match["score"] = float(scores[idx])
        top_results.append(match)

    return top_results
