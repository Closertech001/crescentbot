import streamlit as st
import json
import os
import re
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from rapidfuzz import fuzz, process

# -------------------------
# Dataset Loaders & Helpers
# -------------------------

@st.cache_data(show_spinner=True)
def load_qa_dataset(filepath="data/crescent_qa.json"):
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    questions = [entry["question"] for entry in data]
    return data, questions

@st.cache_data(show_spinner=True)
def load_course_data(filepath="data/course_data.json"):
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

# -------------------------
# Model and Embeddings
# -------------------------

@st.cache_resource(show_spinner=True)
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_data(show_spinner=True)
def compute_question_embeddings(questions):
    model = load_model()
    embeddings = model.encode(questions, convert_to_numpy=True, normalize_embeddings=True)
    return embeddings

def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine similarity via inner product on normalized embeddings
    index.add(embeddings)
    return index

# -------------------------
# Semantic Search Function
# -------------------------

def semantic_search(query, model, index, qa_data, top_k=1):
    q_emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    D, I = index.search(q_emb, top_k)
    best_match = qa_data[I[0][0]]
    score = D[0][0]
    return best_match, score

# -------------------------
# Course Query Utils
# -------------------------

department_faculty_map = {
    "computer science": "CICOT",
    "information technology": "CICOT",
    "mass communication": "CASMAS",
    "accounting": "CASMAS",
    "business administration": "CASMAS",
    "architecture": "COES",
    "nursing": "COHES",
    "physiology": "COHES",
    "anatomy": "COHES",
    "physics": "CONAS",
    "chemistry": "CONAS",
    "microbiology": "CONAS",
    "biochemistry": "CONAS",
    "law": "BACOLAW"
}

def fuzzy_match_department(dept, course_data, threshold=80):
    all_depts = list(course_data.keys())
    match, score = process.extractOne(dept, all_depts, scorer=fuzz.token_sort_ratio)
    return match if score >= threshold else None

def extract_course_info(user_input, course_data, memory):
    dept_match = re.search(r"(?:in|for|of)?\s*(\b[a-zA-Z ]{3,}\b department)?\s*(computer science|mass communication|nursing|law|biochemistry|architecture|accounting|microbiology|anatomy|physiology|physics|chemistry)", user_input, re.IGNORECASE)
    level_match = re.search(r"(\d{3}|\d{2,3}-level|\d{3} level|\d{1,3}00|\d{1}-level)", user_input)
    sem_match = re.search(r"(first|1st|second|2nd)\s+semester", user_input.lower())

    department = dept_match.group(2).strip().lower() if dept_match else memory.get("department")
    level = level_match.group(1) if level_match else memory.get("level")
    semester = sem_match.group(1).lower() if sem_match else memory.get("semester")

    if level:
        level = re.findall(r"\d+", level)[0] + "00"
    if semester:
        semester = "first" if "1" in semester or "first" in semester else "second"

    if department:
        department = fuzzy_match_department(department, course_data)
    else:
        return None

    if not department or department not in course_data:
        return "⚠️ I couldn't find that department."

    memory["department"] = department
    if level: memory["level"] = level
    if semester: memory["semester"] = semester

    dept_data = course_data[department]
    if level not in dept_data:
        return f"📘 No course data found for {level}-level {department.title()}."

    level_data = dept_data[level]

    if semester:
        sem_key = "1st" if "first" in semester else "2nd"
        courses = level_data.get(sem_key, [])
        if not courses:
            return f"📙 No courses found for {semester} semester."
        return f"📘 Courses for {department.title()} {level}-level ({semester} semester):\n" + "\n".join(
            [f"- `{c['code']}`: *{c['title']}* ({c['unit']} units)" for c in courses]
        )
    else:
        courses_1 = level_data.get("1st", [])
        courses_2 = level_data.get("2nd", [])
        response = f"{department.title()} {level}-level courses:\n"
        if courses_1:
            response += "\n📘 First Semester:\n" + "\n".join([f"- `{c['code']}`: *{c['title']}* ({c['unit']} units)" for c in courses_1])
        if courses_2:
            response += "\n\n📗 Second Semester:\n" + "\n".join([f"- `{c['code']}`: *{c['title']}* ({c['unit']} units)" for c in courses_2])
        return response

def get_course_by_code(user_input, course_data):
    code_match = re.search(r"\b([A-Z]{2,4}\s?\d{3})\b", user_input.upper())
    if not code_match:
        return None

    course_code = code_match.group(1).replace(" ", "").upper()
    for dept, levels in course_data.items():
        for level, semesters in levels.items():
            for sem, courses in semesters.items():
                for course in courses:
                    if course["code"].replace(" ", "").upper() == course_code:
                        return f"📘 `{course_code}` is *{course['title']}* offered in `{dept.title()}`, {level}-level ({sem} semester), worth {course['unit']} unit(s)."
    return "⚠️ I couldn't find any course matching that code."

# -------------------------
# Main App
# -------------------------

def main():
    st.set_page_config(page_title="CrescentBot - University Assistant", page_icon="🎓", layout="centered")
    st.title("🤖 CrescentBot – Crescent University Assistant")

    qa_data, questions = load_qa_dataset()
    course_data = load_course_data()
    model = load_model()
    embeddings = compute_question_embeddings(questions)
    index = build_faiss_index(embeddings)

    memory = {}

    query = st.text_input("Ask me anything about Crescent University:")

    if query:
        query_lower = query.lower()

        # Check for course code queries first
        course_response = get_course_by_code(query, course_data)
        if course_response:
            st.markdown(course_response)
            return

        # Check for course info queries
        course_info_response = extract_course_info(query, course_data, memory)
        if course_info_response and course_info_response.startswith("⚠️") is False:
            st.markdown(course_info_response)
            return
        elif course_info_response and course_info_response.startswith("⚠️"):
            # If error about department, just show it
            st.markdown(course_info_response)
            return

        # Otherwise, do semantic search on QA dataset
        best_match, score = semantic_search(query, model, index, qa_data)
        if score > 0.6:
            st.markdown(f"**Answer:** {best_match['answer']}")
        else:
            st.markdown("❓ Sorry, I couldn't find a confident answer. Please try rephrasing your question.")

if __name__ == "__main__":
    main()
