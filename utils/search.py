import torch
from sentence_transformers.util import cos_sim

def find_response(query, model, dataset, embeddings, top_k=3, threshold=0.45):
    """
    Find the best response using semantic similarity.
    
    Args:
        query (str): User's query.
        model: SentenceTransformer model.
        dataset (pd.DataFrame): Dataset with 'question' and 'answer'.
        embeddings (Tensor): Precomputed question embeddings.
        top_k (int): Number of top matches to consider for related questions.
        threshold (float): Minimum similarity score to accept as valid.
    
    Returns:
        Tuple[str, List[str]]: Answer and list of related questions.
    """
    # Encode query
    query_embedding = model.encode([query], convert_to_tensor=True)
    
    # Compute cosine similarity
    similarities = cos_sim(query_embedding, embeddings)[0]
    
    # Get top-k indices
    top_scores, top_indices = torch.topk(similarities, top_k)
    top_score = top_scores[0].item()
    top_index = top_indices[0].item()

    matched_question = dataset.iloc[top_index]["question"]
    matched_answer = dataset.iloc[top_index]["answer"]

    print(f"[DEBUG] Top match: '{matched_question}' (Score: {top_score:.4f})")

    if top_score < threshold:
        return (
            "I'm sorry, I couldn't find an exact answer to that. Try rephrasing your question.",
            []
        )
    
    # Collect related questions (excluding top match)
    related_qs = [
        dataset.iloc[i]["question"]
        for i in top_indices[1:]
        if i != top_index
    ]

    return matched_answer, related_qs
