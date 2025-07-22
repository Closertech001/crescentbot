import json
import pandas as pd
from sentence_transformers import SentenceTransformer
from typing import List

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
