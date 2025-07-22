# utils/embedding.py

import json
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple

# --- Caching the model and dataset to avoid repeated loads ---
_model = None
_df = None
_question_embeddings = None

# --- Load & Cache Model ---
def load_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(model_name)
    return _model

# --- Load & Cache Dataset ---
def load_dataset(path: str = "data/crescent_qa.json") -> pd.DataFrame:
    global _df
    if _df is None:
        try:
            if path.endswith(".jsonl"):
                with open(path, "r", encoding="utf-8") as f:
                    data = [json.loads(line) for line in f]
            else:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            _df = pd.DataFrame(data)
        except Exception as e:
            raise RuntimeError(f"[Dataset Load Error] Could not load file at {path}: {e}")
    return _df

# --- Compute Embeddings for Questions ---
def compute_question_embeddings(questions: List[str], model: SentenceTransformer):
    return model.encode([q.strip().lower() for q in questions], convert_to_tensor=True)

# --- Top-k Semantic Search ---
def get_top_k_answers(query: str, top_k: int = 3) -> List[Tuple[str, float]]:
    global _question_embeddings
    model = load_model()
    df = load_dataset()
    questions = df["question"].tolist()
    answers = df["answer"].tolist()

    # Cache question embeddings
    if _question_embeddings is None:
        _question_embeddings = compute_question_embeddings(questions, model)

    # Encode query and calculate similarity
    query_embedding = model.encode([query.strip().lower()], convert_to_tensor=True)
    scores = cosine_similarity(query_embedding, _question_embeddings)[0]
    top_indices = np.argsort(scores)[::-1][:top_k]

    return [(answers[i], float(scores[i])) for i in top_indices]
