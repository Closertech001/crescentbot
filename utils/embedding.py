from sentence_transformers import SentenceTransformer, util
import json

def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

def load_dataset():
    with open("data/crescent_qa.json", "r", encoding="utf-8") as f:
        return json.load(f)

def compute_question_embeddings(questions, model):
    return model.encode(questions, convert_to_tensor=True)
