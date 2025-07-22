import torch
from typing import List, Tuple, Dict, Union
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
import pandas as pd


def find_response(
    query: str,
    model: SentenceTransformer,
    dataset: pd.DataFrame,
    embeddings: torch.Tensor,
    top_k: int = 3,
    threshold: float = 0.45,
    debug: bool = False
) -> Tuple[str, List[str]]:
    """
    Find the best response to a user query using cosine similarity on embeddings.

    Returns:
        Tuple[str, List[str]]: Best matching answer and list of related questions.
    """
    # Encode user query
    query_embedding = model.encode([query.strip().lower()], convert_to_tensor=True)

    # Compute cosine similarity between query and precomputed question embeddings
    similarities = cos_sim(query_embedding, embeddings)[0]

    # Get top-k most similar indices and scores
    top_scores, top_indices = torch.topk(similarities, top_k)

    top_score = top_scores[0].item()
    top_index = top_indices[0].item()

    if debug:
        matched_question = dataset.iloc[top_index]["question"]
        print(f"[DEBUG] Top match: '{matched_question}' (Score: {top_score:.4f})")

    # Check similarity threshold
    if top_score < threshold:
        return (
            "I'm sorry, I couldn't find an exact answer to that. Try rephrasing your question.",
            []
        )

    # Collect related questions (excluding the top match)
    related_qs = [
        dataset.iloc[i]["question"]
        for i in top_indices[1:]
        if i != top_index and i < len(dataset)
    ]

    # Return the top match answer
    return dataset.iloc[top_index]["answer"], related_qs


def search_similar(
    query: str,
    df: pd.DataFrame,
    embeddings: torch.Tensor,
    model: SentenceTransformer,
    top_k: int = 1
) -> Dict[str, Union[str, float]]:
    """
    Return the best-matching QA pair and similarity score.
    Useful as a lightweight search function.
    """
    query_embedding = model.encode(query.lower().strip(), convert_to_tensor=True)
    similarities = cos_sim(query_embedding, embeddings)[0]
    top_scores, top_indices = torch.topk(similarities, top_k)

    top_index = int(top_indices[0])
    score = float(top_scores[0])

    return {
        "question": df.iloc[top_index]["question"],
        "answer": df.iloc[top_index]["answer"],
        "score": score
    }
