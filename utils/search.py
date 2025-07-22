# utils/search.py

from utils.embedding import load_model_and_embeddings
import numpy as np

def semantic_search(query: str, top_k: int = 3, threshold: float = 0.6):
    model, data, index, _ = load_model_and_embeddings()

    query_vec = model.encode([query.strip().lower()])
    D, I = index.search(np.array(query_vec), top_k)

    top_score = D[0][0]
    top_index = I[0][0]

    if top_score > threshold:
        return "🤖 Sorry, I couldn’t find a close match. Please try rephrasing your question.", []

    best_answer = data[top_index]["answer"]
    related_questions = [data[i]["question"] for i in I[0][1:] if i != top_index]

    return best_answer, related_questions
