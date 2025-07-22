import json
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple

# --- Model & Embedding Utilities ---

def load_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    """Load the SentenceTransformer model."""
    return SentenceTransformer(model_name)

def load_dataset(path: str = "data/crescent_qa.json") -> pd.DataFrame:
    """Load a JSON or JSONL QA dataset and return as a pandas DataFrame."""
    try:
        if path.endswith(".jsonl"):
            with open(path, "r", encoding="utf-8") as f:
                data = [json.loads(line) for line in f]
        else:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        return pd.DataFrame(data)
    except Exception as e:
        raise RuntimeError(f"[Dataset Load Error] Could not load file at {path}: {e}")

def compute_question_embeddings(questions: List[str], model: SentenceTransformer):
    """Compute embeddings for a list of questions using the provided model."""
    return model.encode([q.strip().lower() for q in questions], convert_to_tensor=True)

# --- Main Top-k Search Function ---

def get_top_k_answers(query: str, top_k: int = 3) -> List[Tuple[str, float]]:
    """Return top-k matching answers from the QA dataset based on semantic similarity."""
    model = load_model()
    df = load_dataset()
    questions = df["question"].tolist()
    answers = df["answer"].tolist()

    # Compute embeddings
    question_embeddings = compute_question_embeddings(questions, model)
    query_embedding = model.encode([query.strip().lower()], convert_to_tensor=True)

    # Compute similarity scores
    scores = cosine_similarity(query_embedding, question_embeddings)[0]
    top_indices = np.argsort(scores)[::-1][:top_k]

    # Return answers with scores
    return [(answers[i], float(scores[i])) for i in top_indices]
