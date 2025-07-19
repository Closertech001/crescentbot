import streamlit as st
import os
import uuid
import openai
from dotenv import load_dotenv

from utils.embedding import load_model, load_data, compute_question_embeddings
from utils.preprocess import preprocess_text
from utils.search import find_response
from utils.memory import init_memory
from utils.log_utils import log_query

# --- Load Environment Variables ---
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# --- Page Settings ---
st.set_page_config(page_title="Crescent University Chatbot", page_icon="🎓")

# --- Ensure Session State is Initialized ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "related_questions" not in st.session_state:
    st.session_state.related_questions = []

if "last_department" not in st.session_state:
    st.session_state.last_department = None

# --- Load Model & Dataset ---
@st.cache_resource
def load_bot_resources():
    model = load_model()
    data = load_data()
    return model, data

model, dataset = load_bot_resources()
question_embeddings = compute_question_embeddings(dataset["question"].tolist(), model)

# --- Sidebar ---
with st.sidebar:
    st.markdown("### 💬 CrescentBot")
    if st.button("🧹 Clear Chat"):
        st.session_state.chat_history = []
        st.session_state.related_questions = []
        st.session_state.last_department = None
        st.rerun()

# --- Styles ---
st.markdown("""
<style>
    html, body, .stApp { font-family: 'Segoe UI', sans-serif; }
    h1 { color: #004080; }
    .chat-message-user {
        background-color: #d6eaff;
        padding: 12px;
        border-radius: 15px;
        margin-bottom: 10px;
        margin-left: auto;
        max-width: 75%;
        font-weight: 550;
        color: #000;
    }
    .chat-message-assistant {
        background-color: #f5f5f5;
        padding: 12px;
        border-radius: 15px;
        margin-bottom: 10px;
        margin-right: auto;
        max-width: 75%;
        font-weight: 600;
        color: #000;
    }
    .related-question {
        background-color: #e6f2ff;
        padding: 8px 12px;
        margin: 6px 6px 6px 0;
        display: inline-block;
        border-radius: 10px;
        font-size: 0.9rem;
        cursor: pointer;
    }
    .department-label {
        font-size: 0.8rem;
        color: #004080;
        font-style: italic;
    }
</style>
""", unsafe_allow_html=True)

# --- Title ---
st.title("🎓 Crescent University Chatbot")

# --- Render Chat History ---
for msg in st.session_state.chat_history:
    css_class = "chat-message-user" if msg["role"] == "user" else "chat-message-assistant"
    with st.chat_message(msg["role"]):
        st.markdown(f'<div class="{css_class}">{msg["content"]}</div>', unsafe_allow_html=True)
        if msg["role"] == "assistant" and st.session_state.last_department:
            st.markdown(f'<div class="department-label">Department: {st.session_state.last_department}</div>', unsafe_allow_html=True)

# --- User Input ---
user_input = st.chat_input("Ask me anything about Crescent University...")

if user_input:
    cleaned_input = preprocess_text(user_input)
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    # --- Check for exact match ---
    matched_row = dataset[dataset['question'].str.lower() == cleaned_input.lower()]
    if not matched_row.empty:
        response = matched_row.iloc[0]['answer']
        department = None
        related = []
        score = 1.0
    else:
        response, department, score, related = find_response(cleaned_input, dataset, question_embeddings)

    st.session_state.chat_history.append({"role": "assistant", "content": response})
    st.session_state.related_questions = related
    st.session_state.last_department = department

    log_query(user_input, score)
    st.rerun()

# --- Suggested Follow-up Questions ---
if st.session_state.related_questions:
    st.markdown("#### 💡 You might also ask:")
    for i, q in enumerate(st.session_state.related_questions):
        if st.button(q, key=f"related_{i}", use_container_width=True):
            st.session_state.chat_history.append({"role": "user", "content": q})
            response, department, score, related = find_response(q, dataset, question_embeddings)
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            st.session_state.related_questions = related
            st.session_state.last_department = department
            log_query(q, score)
            st.rerun()
