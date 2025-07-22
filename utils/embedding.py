# utils/embedding.py

import json
import os
from sentence_transformers import SentenceTransformer
import numpy as np

# Load QA dataset and compute embeddings for semantic search
MODEL_NAME = "all-MiniLM-L6-v2"
MODEL = SentenceTransformer(MODEL_NAME)

EMBEDDINGS_FILE = "data/qa_embeddings.json"
QA_FILE = "data/crescent_qa.json"


def load_qa_data():
    with open(QA_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_embeddings(questions):
    return MODEL.encode(questions, convert_to_numpy=True, normalize_embeddings=True).tolist()


def get_stored_embeddings():
    if os.path.exists(EMBEDDINGS_FILE):
        with open(EMBEDDINGS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        qa_data = load_qa_data()
        questions = [item["question"] for item in qa_data]
        embeddings = compute_embeddings(questions)
        with open(EMBEDDINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(embeddings, f)
        return embeddings
