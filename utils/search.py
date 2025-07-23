def find_response(user_query, dataset, embeddings, model=None, threshold=0.6):
    from sentence_transformers.util import cos_sim
    import torch

    if model is None:
        from .embedding import load_model
        model = load_model()

    query_embedding = model.encode(user_query, convert_to_tensor=True)
    cosine_scores = cos_sim(query_embedding, embeddings)[0]

    top_idx = torch.argmax(cosine_scores).item()
    top_score = cosine_scores[top_idx].item()

    if top_score < threshold:
        return "😕 I’m not sure how to answer that.", None, top_score, []

    best_row = dataset.iloc[top_idx]
    response = best_row["answer"]
    department = best_row.get("department", None)

    # Optional: fetch top 3 related questions
    top_k = torch.topk(cosine_scores, k=4)
    top_related = []
    for idx in top_k.indices.tolist():
        if idx != top_idx:
            top_related.append(dataset.iloc[idx]["question"])

    return response, department, top_score, top_related
