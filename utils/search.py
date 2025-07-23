# utils/search.py

from sentence_transformers.util import cos_sim
import torch

def semantic_search(query, model, embeddings, df, threshold=0.6):
    """
    Perform semantic similarity search between user query and dataset questions.
    
    Parameters:
    - query (str): User input
    - model (SentenceTransformer): Loaded embedding model
    - embeddings (Tensor): Precomputed embeddings for all questions
    - df (DataFrame): QA dataframe with a 'question' and 'answer' column
    - threshold (float): Minimum cosine similarity to accept a match
    
    Returns:
    - str or None: Matched answer or None if below threshold
    """
    user_embedding = model.encode(query.strip().lower(), convert_to_tensor=True)
    cosine_scores = cos_sim(user_embedding, embeddings)[0]
    best_score = torch.max(cosine_scores).item()
    best_idx = torch.argmax(cosine_scores).item()
    
    if best_score >= threshold:
        return df.iloc[best_idx]["answer"]
    return None
