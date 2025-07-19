from sentence_transformers import util
from utils.preprocess import preprocess_text
import random

def find_response(query, dataset, embeddings):
    from utils.embedding import compute_question_embeddings, load_model
    model = load_model()
    processed = preprocess_text(query)
    query_embedding = compute_question_embeddings([processed], model)[0]
    
    scores = util.pytorch_cos_sim(query_embedding, embeddings)[0]
    top_k = scores.topk(5)
    best_idx = top_k.indices[0].item()
    score = top_k.values[0].item()

    response = dataset["answer"][best_idx]
    related = [dataset["question"][i] for i in top_k.indices.tolist()[1:]]

    department = dataset.get("department", [""] * len(dataset["question"]))[best_idx]
    return response, department, score, related
