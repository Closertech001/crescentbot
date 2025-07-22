# utils/embedding.py

import json
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import torch

def load_dataset(path="data/crescent_qa.json"):
    """Loads the Q&A dataset from JSON file into a DataFrame."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return pd.DataFrame(data)

def compute_question_embeddings(questions, model):
    """Computes sentence embeddings for all questions."""
    with torch.no_grad():
        embeddings = model.encode(
            [q.strip().lower() for q in questions],
            convert_to_tensor=True,
            normalize_embeddings=True
        )
    return embeddings
