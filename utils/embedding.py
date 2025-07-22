import json
import os
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer
import numpy as np

# ✅ Use Streamlit cache to avoid re-downloading every run
@st.cache_resource(show_spinner="🔍 Loading embedding model...")
def load_model(model_name="all-MiniLM-L6-v2"):
    return SentenceTransformer(model_name)

# ✅ Load the model once globally when this module is imported
model = load_model()

# Load QA data and compute embeddings
@st.cache_data(show_spinner="🔎 Computing QA embeddings...")
def load_qa_embeddings(qa_json_path):
    with open(qa_json_path, "r", encoding="utf-8") as f:
        qa_data = json.load(f)

    questions = [item["question"] for item in qa_data]
    embeddings = model.encode(questions, show_progress_bar=True)
    return qa_data, np.array(embeddings)
