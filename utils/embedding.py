import json
import os
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

def load_embeddings(json_path="data/crescent_qa.json"):
    with open(json_path, "r", encoding="utf-8") as f:
        qa_data = json.load(f)

    questions = [item["question"] for item in qa_data]
    answers = [item["answer"] for item in qa_data]
    embeddings = model.encode(questions, convert_to_tensor=False)

    return list(zip(questions, answers, embeddings))
