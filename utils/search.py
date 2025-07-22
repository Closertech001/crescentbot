import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def search_similar(query, df, embeddings, model, top_k=3, threshold=0.4):
    """
    Search top_k similar questions from the dataset based on cosine similarity.

    Args:
        query (str): The user query.
        df (DataFrame): The QA dataset.
        embeddings (np.array): Precomputed embeddings of questions.
        model (SentenceTransformer): The embedding model.
        top_k (int): Number of top matches to return.
        threshold (float): Minimum similarity score to consider.

    Returns:
        List of dicts with 'question', 'answer', and 'score'.
    """
    query_embedding = model.encode([query], convert_to_numpy=True)
    scores = cosine_similarity(query_embedding, embeddings)[0]
    
    # Get top_k results above the threshold
    ranked_indices = np.argsort(scores)[::-1]
    results = []
    for idx in ranked_indices[:top_k]:
        if scores[idx] >= threshold:
            results.append({
                "question": df.iloc[idx]["question"],
                "answer": df.iloc[idx]["answer"],
                "score": float(scores[idx])
            })

    return results
