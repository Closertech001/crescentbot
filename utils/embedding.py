import json
import pandas as pd
from sentence_transformers import SentenceTransformer

# 🔧 Load SentenceTransformer model
def load_model(model_name="all-MiniLM-L6-v2"):
    """Load the SentenceTransformer model."""
    return SentenceTransformer(model_name)

# 📂 Load dataset
def load_dataset(path="data/crescent_qa.json"):
    """Load the dataset from JSON or JSONL and return as a DataFrame."""
    try:
        if path.endswith(".jsonl"):
            with open(path, "r", encoding="utf-8") as f:
                data = [json.loads(line) for line in f]
        else:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        return pd.DataFrame(data)
    except Exception as e:
        raise RuntimeError(f"[Dataset Load Error] Could not load file at {path}: {e}")

# 📐 Compute embeddings
def compute_question_embeddings(questions, model):
    """Compute embeddings for a list of questions using the provided model."""
    cleaned = [q.strip().lower() for q in questions]
    return model.encode(cleaned, convert_to_tensor=True)
