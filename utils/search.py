import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def get_top_k_matches(user_query_embedding, question_embeddings, qa_data, k=3, threshold=0.55):
    """
    Get the top-k most similar questions to the user query based on cosine similarity.
    Returns a list of matching (question, answer, score).
    """
    if isinstance(user_query_embedding, list):
        user_query_embedding = np.array(user_query_embedding)

    scores = cosine_similarity([user_query_embedding], question_embeddings)[0]
    top_indices = np.argsort(scores)[::-1]  # Sort descending

    results = []
    for idx in top_indices[:k]:
        if scores[idx] >= threshold:
            q = qa_data[idx]["question"]
            a = qa_data[idx]["answer"]
            score = round(float(scores[idx]), 3)
            results.append({"question": q, "answer": a, "score": score})

    return results
