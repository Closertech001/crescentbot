import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

def load_dataset(filepath="data/crescent_qa.json"):
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

def compute_question_embeddings(questions, model):
    return model.encode(questions, convert_to_numpy=True, normalize_embeddings=True)

def load_model(model_name="all-MiniLM-L6-v2"):
    return SentenceTransformer(model_name)

def build_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # cosine similarity
    index.add(embeddings)
    return index
