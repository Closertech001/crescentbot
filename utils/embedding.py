import json
from sentence_transformers import SentenceTransformer

def load_model(model_name="all-MiniLM-L6-v2"):
    """Load the sentence transformer model for encoding questions."""
    return SentenceTransformer(model_name)

def load_qa_dataset(filepath="data/crescent_qa.json"):
    """Load the Q&A pairs from a JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def compute_question_embeddings(model, qa_data):
    """Compute embeddings for all questions in the dataset."""
    questions = [item["question"] for item in qa_data]
    embeddings = model.encode(questions, show_progress_bar=True)
    return embeddings
