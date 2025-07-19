# utils/embedding.py

import json
import pandas as pd
from sentence_transformers import SentenceTransformer

def load_model(model_name="all-MiniLM-L6-v2"):
    """Load the SentenceTransformer model."""
    return SentenceTransformer(model_name)

def load_dataset(path="data/crescent_qa.json"):
    """Load the dataset from JSON and return as a DataFrame."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return pd.DataFrame(data)
    except Exception as e:
        raise RuntimeError(f"[Dataset Load Error] Could not load file at {path}: {e}")

def compute_question_embeddings(questions, model):
    """Compute embeddings for a list of questions using the provided model."""
    return model.encode(questions, convert_to_tensor=True)
