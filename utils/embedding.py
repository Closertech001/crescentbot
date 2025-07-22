import json
import numpy as np
from sentence_transformers import SentenceTransformer

def load_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    """
    Load and return a SentenceTransformer model.
    
    Args:
        model_name (str): Name of the pretrained model to load.

    Returns:
        SentenceTransformer: Loaded sentence embedding model.
    """
    return SentenceTransformer(model_name)

def load_qa_data(qa_path: str = "data/crescent_qa.json") -> list:
    """
    Load Q&A pairs from a JSON file.

    Args:
        qa_path (str): Path to the Q&A JSON file.

    Returns:
        list: List of dictionaries with 'question' and 'answer' keys.
    """
    with open(qa_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def get_question_embeddings(questions: list, model: SentenceTransformer) -> np.ndarray:
    """
    Generate normalized embeddings for a list of questions.

    Args:
        questions (list): List of question strings.
        model (SentenceTransformer): Preloaded embedding model.

    Returns:
        np.ndarray: Normalized question embeddings (2D array).
    """
    return model.encode(
        questions,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
