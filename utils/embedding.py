import pandas as pd
import json
from sentence_transformers import SentenceTransformer
import streamlit as st

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_data
def load_data(path="qa_dataset.json"):
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    clean = []
    for entry in raw:
        if entry.get("question") and entry.get("answer"):
            clean.append({
                "text": f"Q: {entry['question'].strip()}\nA: {entry['answer'].strip()}",
                "question": entry["question"].strip(),
                "answer": entry["answer"].strip(),
                "department": entry.get("department", "").strip(),
                "level": entry.get("level", "").strip(),
                "semester": entry.get("semester", "").strip(),
                "faculty": entry.get("faculty", "").strip()
            })

    return pd.DataFrame(clean)

@st.cache_data
def compute_question_embeddings(questions):
    model = load_model()
    return model.encode(questions, convert_to_tensor=True)
