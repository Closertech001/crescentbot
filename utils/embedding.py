# utils/embedding.py

import json
import numpy as np
from sentence_transformers import SentenceTransformer
import streamlit as st

@st.cache_resource
def load_model(model_name="all-MiniLM-L6-v2"):
    """
    Load and cache the SentenceTransformer model for CPU usage.
    Optimized for low memory environments like Streamlit Cloud.
    """
    return SentenceTransformer(model_name, device="cpu")

@st.cache_data
def load_qa_data(qa_path="data/crescent_qa.json"):
    """
    Load and cache the Q&A dataset from a JSON file.
    """
    with open(qa_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

@st.cache_data
def compute_question_embeddings(questions, model):
    """
    Compute and normalize embeddings for a list of questions.
    Returns a numpy array of shape (len(questions), embedding_dim).
    """
    embeddings = model.encode(
        questions,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )
    return embeddings
