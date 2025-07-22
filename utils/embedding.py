import json
import pandas as pd
from sentence_transformers import SentenceTransformer
import torch
import os

# Cache global state
_model = None
_dataset = None
_embeddings = None


def load_model(model_name="all-MiniLM-L6-v2"):
    """
    Load and return the sentence transformer model.
    """
    global _model
    if _model is None:
        _model = SentenceTransformer(model_name)
    return _model


def load_dataset(path="data/crescent_qa.json"):
    """
    Load the question-answer dataset from JSON file.
    """
    global _dataset
    if _dataset is None:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        _dataset = pd.DataFrame(data)
    return _dataset


def compute_embeddings(model, dataset):
    """
    Compute and return embeddings for the questions.
    """
    global _embeddings
    if _embeddings is None:
        questions = dataset["question"].str.lower().tolist()
        _embeddings = model.encode(questions, convert_to_tensor=True)
    return _embeddings


def load_model_and_embeddings():
    """
    Load model, dataset, and compute embeddings. Returns all three.
    """
    model = load_model()
    dataset = load_dataset()
    embeddings = compute_embeddings(model, dataset)
    return model, dataset, embeddings


def get_top_k_answer(query, top_k=3, threshold=0.45):
    """
    Wrapper to return best-matched answer and related questions.
    """
    model, dataset, embeddings = load_model_and_embeddings()
    from utils.search import find_response  # imported here to avoid circular import
    return find_response(query, model, dataset, embeddings, top_k, threshold)
