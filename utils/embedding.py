import json
import pandas as pd
from sentence_transformers import SentenceTransformer

def load_model(model_name="all-MiniLM-L6-v2"):
    """Load a sentence transformer model for embeddings."""
    return SentenceTransformer(model_name)

def load_qa_data(path="data/crescent_qa.json"):
    """Load Q&A dataset from JSON."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return pd.DataFrame(data)

def compute_embeddings(model, questions):
    """Compute embeddings for all questions."""
    return model.encode(questions, show_progress_bar=True, convert_to_tensor=True)
