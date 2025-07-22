import json
import pandas as pd
from sentence_transformers import SentenceTransformer

def load_model(model_name="all-MiniLM-L6-v2"):
    return SentenceTransformer(model_name)

def load_dataset(path="data/crescent_qa.json"):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return pd.DataFrame(data)

def compute_question_embeddings(questions, model):
    return model.encode(questions, show_progress_bar=True, convert_to_tensor=True)
