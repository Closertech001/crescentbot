import json
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import os

def load_dataset(path):
    """Load JSON dataset and convert to DataFrame."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"[ERROR] QA dataset not found at: {path}")
    
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Validate structure
    if not isinstance(data, list) or not all("question" in item and "answer" in item for item in data):
        raise ValueError("[ERROR] Invalid format: QA dataset must be a list of {question, answer} dictionaries.")

    print(f"[INFO] Loaded {len(data)} QA pairs from {path}")
    return pd.DataFrame(data)

def compute_question_embeddings(questions, model):
    """Compute embeddings for a list of questions using the given SentenceTransformer model."""
    if not questions:
        raise ValueError("[ERROR] No questions provided to compute embeddings.")
    
    print(f"[INFO] Computing embeddings for {len(questions)} questions...")
    embeddings = model.encode(questions, convert_to_tensor=True)
    return embeddings

def load_qa_embeddings(qa_path="data/crescent_qa.json", model_name="all-MiniLM-L6-v2"):
    """Load QA data and compute embeddings."""
    model = SentenceTransformer(model_name)
    df = load_dataset(qa_path)
    questions = df["question"].tolist()
    embeddings = compute_question_embeddings(questions, model)
    return df.to_dict(orient="records"), embeddings
