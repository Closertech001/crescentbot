# utils/search.py

from utils.embedding import load_model_and_embeddings
import numpy as np

def semantic_search(query: str, top_k: int = 3, threshold: float = 0.45):
    model, data, index, embeddings = load_model_and_embeddings()

    # Embed the query
    query_vec = model.encode([query.strip().lower()])
    D, I = index.search(np.array(query_vec), top_k)  # distances, indices

    top_score = D[0][0]
    top_index = I[0][0]

    if top_score > threshold:
        return (
            "I'm sorry, I couldn't find an exact answer to that. Try rephrasing your question. 🤔",
            []
        )

    best_answer = data[top_index]["answer"]
    related_qs = [data[i]["question"] for i in I[0][1:] if i != top_index]

    print(f"[DEBUG] FAISS Match Score: {top_score:.4f}")

    return best_answer, related_qs
