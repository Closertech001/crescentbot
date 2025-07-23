import torch
from sentence_transformers.util import cos_sim

def find_response(user_question, model, embeddings, df, threshold=0.6):
    user_embedding = model.encode(user_question.strip().lower(), convert_to_tensor=True)
    cosine_scores = cos_sim(user_embedding, embeddings)[0]
    best_score = torch.max(cosine_scores).item()
    best_idx = torch.argmax(cosine_scores).item()

    if best_score >= threshold:
        return df.iloc[best_idx]["answer"]
    return None
