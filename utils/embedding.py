# utils/embedding.py

import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import streamlit as st

@st.cache_data(show_spinner="Loading model and embedding index...")
def load_model_and_embeddings(json_path="data/crescent_qa.json"):
    model = SentenceTransformer("all-MiniLM-L6-v2")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    questions = [item["question"].strip().lower() for item in data]
    embeddings = model.encode(questions, convert_to_numpy=True)

    # Create FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    return model, data, index, embeddings
