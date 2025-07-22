from sentence_transformers import util

def search_similar(query, df, embeddings, model, top_k=1):
    """Find the most similar QA pair using semantic search."""
    query_emb = model.encode(query.lower().strip(), convert_to_tensor=True)
    scores = util.pytorch_cos_sim(query_emb, embeddings)[0]
    top_results = scores.topk(top_k)

    idx = int(top_results.indices[0])
    score = float(top_results.values[0])

    return {
        "question": df.iloc[idx]["question"],
        "answer": df.iloc[idx]["answer"],
        "score": score
    }
