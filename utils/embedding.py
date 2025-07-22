import json
import pandas as pd
from sentence_transformers import SentenceTransformer

def load_model(model_name="all-MiniLM-L6-v2"):
    """Loads SentenceTransformer model for embedding."""
    return SentenceTransformer(model_name)

def load_qa_dataset(path="data/crescent_qa.json"):
    """Loads QA dataset from JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def compute_embeddings(model, qa_data):
    """Generates embeddings for all questions in the QA dataset."""
    questions = [entry["question"] for entry in qa_data]
    embeddings = model.encode(questions, show_progress_bar=True)
    return embeddings
