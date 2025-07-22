# utils/embedding.py

import json
import os
from sentence_transformers import SentenceTransformer, util
import numpy as np

# Load QA dataset and compute embeddings for semantic search
MODEL_NAME = "all-MiniLM-L6-v2"
MODEL = SentenceTransformer(MODEL_NAME)

EMBEDDINGS_FILE = "data/qa_embeddings.json"
QA_FILE = "data/crescent_qa.json"


def load_qa_data(filepath=QA_FILE):
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_embeddings(questions):
    return MODEL.encode(questions, convert_to_numpy=True, normalize_embeddings=True)


def load_qa_embeddings(qa_path=QA_FILE, emb_path=EMBEDDINGS_FILE):
    qa_data = load_qa_data(qa_path)
    questions = [item["question"] for item in qa_data]

    if os.path.exists(emb_path):
        with open(emb_path, "r", encoding="utf-8") as f:
            embeddings = np.array(json.load(f))
    else:
        embeddings = compute_embeddings(questions)
        with open(emb_path, "w", encoding="utf-8") as f:
            json.dump(embeddings.tolist(), f)

    return qa_data, embeddings


def find_most_similar_question(query, qa_data, embeddings, top_k=1):
    query_embedding = compute_embeddings([query])[0]
    scores = util.cos_sim(query_embedding, embeddings)[0]
    best_idx = int(np.argmax(scores))
    best_score = float(scores[best_idx])
    return qa_data[best_idx], best_score
