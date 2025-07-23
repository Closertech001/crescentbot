import json
import pandas as pd
from sentence_transformers import SentenceTransformer
import torch

def load_model(model_name="all-MiniLM-L6-v2"):
    return SentenceTransformer(model_name)

def load_dataset(path="data/crescent_qa.json"):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return pd.DataFrame(data)

# This alias is added in case some other file tries to import `load_data`
load_data = load_dataset

def compute_question_embeddings(questions, model):
    embeddings = model.encode(questions, convert_to_tensor=True, show_progress_bar=True)
    return embeddings
