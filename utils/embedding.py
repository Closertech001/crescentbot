import pandas as pd
import json
from sentence_transformers import SentenceTransformer
import streamlit as st

# --- Load embedding model ---
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

# --- Load and clean dataset ---
@st.cache_data
def load_data(path="qa_dataset.json"):
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    clean = []
    for entry in raw:
        if entry.get("question") and entry.get("answer"):
            clean.append({
                "text": f"Q: {entry['question'].strip()}\nA: {entry['answer'].strip()}",
                "question": entry['question'].strip(),
                "answer": entry['answer'].strip(),
                "department": (entry.get("department") or "").strip(),
                "level": (entry.get("level") or "").strip(),
                "semester": (entry.get("semester") or "").strip(),
                "faculty": (entry.get("faculty") or "").strip()
            })
    return pd.DataFrame(clean)

# --- Compute embeddings with model passed in ---
@st.cache_data
def compute_question_embeddings(questions, model):
    return model.encode(questions, convert_to_tensor=True)
