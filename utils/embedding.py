# utils/embedding.py

import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

def load_model(model_name="all-MiniLM-L6-v2"):
    return SentenceTransformer(model_name)

def load_qa_data(qa_path="data/crescent_qa.json"):
    with open(qa_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def get_question_embeddings(questions, model):
    return model.encode(questions, convert_to_numpy=True, normalize_embeddings=True)

def build_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity (assumes normalized)
    index.add(embeddings)
    return index
