import json
from sentence_transformers import SentenceTransformer
import numpy as np

def load_model(model_name="all-MiniLM-L6-v2"):
    """
    Load the SentenceTransformer model.
    """
    return SentenceTransformer(model_name)

def load_qa_data(qa_path="data/crescent_qa.json"):
    """
    Load the Q&A dataset from a JSON file.
    """
    with open(qa_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def get_question_embeddings(questions, model):
    """
    Convert a list of questions to embeddings.
    """
    return model.encode(questions, convert_to_numpy=True, normalize_embeddings=True)
