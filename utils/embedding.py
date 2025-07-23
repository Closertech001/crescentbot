import json
import pandas as pd
from sentence_transformers import SentenceTransformer

def load_model(model_name="all-MiniLM-L6-v2"):
    """
    Load a SentenceTransformer model (used for embedding user queries and QA data).
    """
    return SentenceTransformer(model_name)

def load_dataset(json_path="data/crescent_qa.json"):
    """
    Load the crescent_qa.json file as a pandas DataFrame.
    Each item in the file should have "question" and "answer".
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return pd.DataFrame(data)

def compute_question_embeddings(questions, model):
    """
    Convert a list of questions into dense vector embeddings.
    """
    return model.encode(questions, convert_to_tensor=True)
