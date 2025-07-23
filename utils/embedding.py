# utils/embedding.py

import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer

# Load sentence transformer model
def load_model(model_name="all-MiniLM-L6-v2"):
    return SentenceTransformer(model_name)

# Load questions and answers from JSON
def load_qa_dataset(path="data/crescent_qa.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# Compute and return embeddings for all questions
def load_embeddings(model=None, path="data/embeddings.npy", qa_path="data/crescent_qa.json"):
    if os.path.exists(path):
        return np.load(path), load_qa_dataset(qa_path)

    if model is None:
        model = load_model()

    qa_data = load_qa_dataset(qa_path)
    questions = [item["question"] for item in qa_data]
    embeddings = model.encode(questions, convert_to_numpy=True)

    # Save embeddings
    np.save(path, embeddings)
    return embeddings, qa_data
