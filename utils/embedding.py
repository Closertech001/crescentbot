# utils/embedding.py

import json
import numpy as np
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')

def load_qa_embeddings(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        qa_data = json.load(f)
    questions = [item["question"] for item in qa_data]
    embeddings = model.encode(questions, convert_to_tensor=True)
    return qa_data, embeddings

def find_most_similar_question(query, qa_data, embeddings):
    query_embedding = model.encode(query, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(query_embedding, embeddings)[0]
    best_match_idx = int(np.argmax(scores))
    return qa_data[best_match_idx], float(scores[best_match_idx])
