import torch
from sentence_transformers.util import cos_sim
from utils.embedding import load_model_and_embeddings

def semantic_search(query: str, top_k: int = 3, threshold: float = 0.45):
    """
    Unified function to perform semantic search compatible with app.py

    Args:
        query (str): The user's input query.
        top_k (int): Number of top answers to consider.
        threshold (float): Minimum similarity threshold for match.

    Returns:
        Tuple[str, List[str]]: Best answer and list of related questions.
    """
    model, df, embeddings = load_model_and_embeddings()

    # Encode query
    query_embedding = model.encode([query.strip().lower()], convert_to_tensor=True)
    similarities = cos_sim(query_embedding, embeddings)[0]

    # Get top-k
    top_scores, top_indices = torch.topk(similarities, top_k)
    top_score = top_scores[0].item()
    top_index = top_indices[0].item()

    best_match = df.iloc[top_index]["question"]
    best_answer = df.iloc[top_index]["answer"]

    print(f"[DEBUG] Best match: '{best_match}' (Score: {top_score:.4f})")

    if top_score < threshold:
        return (
            "I'm sorry, I couldn't find an exact answer to that. Try rephrasing your question. 🤔",
            []
        )

    related_qs = [
        df.iloc[i]["question"]
        for i in top_indices[1:]
        if int(i) != top_index
    ]

    return best_answer, related_qs
