# utils/embedding.py

import json
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

# Global cache
_model = None
_df = None
_embeddings = None

def load_model(model_name="all-MiniLM-L6-v2"):
    """
    Load the SentenceTransformer model.
    """
    return SentenceTransformer(model_name)

def load_qa_data(json_path="data/crescent_qa.json"):
    """
    Load the QA dataset as a pandas DataFrame.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return pd.DataFrame(data)

def compute_embeddings(df, model):
    """
    Compute sentence embeddings from the 'question' column of the dataset.
    """
    questions = df["question"].str.lower().tolist()
    return model.encode(questions, convert_to_tensor=True)

def load_model_and_embeddings():
    """
    Load model, dataset, and precompute embeddings (only once).
    """
    global _model, _df, _embeddings

    if _model is None or _df is None or _embeddings is None:
        _model = load_model()
        _df = load_qa_data()
        _embeddings = compute_embeddings(_df, _model)

    return _model, _df, _embeddings
