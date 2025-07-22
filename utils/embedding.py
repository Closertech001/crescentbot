# utils/embedding.py

import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

def load_model(model_name="all-MiniLM-L6-v2"):
    return SentenceTransformer(model_name)

def load_dataset(filepath="data/crescent_qa.json"):
    """
    Loads the QA dataset and returns both the data and extracted questions.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    questions = [entry["question"] for entry in data]
    return data, questions

def get_question_embeddings(questions, model):
    """
    Encodes a list of questions into normalized embeddings.
    """
    return model.encode(questions, convert_to_numpy=True, normalize_embeddings=True)

def build_faiss_index(embeddings):
    """
    Builds a FAISS index from the question embeddings using cosine similarity.
    """
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
    index.add(embeddings)
    return index
