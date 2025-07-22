import streamlit as st
import json
from openai import OpenAI
from utils.embedding import load_embedding_model, build_faiss_index
from utils.preprocess import normalize_input
from utils.course_query import parse_query, get_courses_for_query
from utils.greetings import is_greeting, get_greeting_response
from utils.memory import save_last_query_info, get_last_query_info
import random

# Load assets
with open("data/crescent_qa.json") as f:
    qa_data = json.load(f)

with open("data/course_data.json") as f:
    course_data = json.load(f)

embedding_model = load_embedding_model()
faiss_index, questions = build_faiss_index(qa_data, embedding_model)

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

st.set_page_config(page_title="Crescent Uni Chatbot", layout="wide")
st.title("🎓 Crescent University Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

user_input = st.chat_input("Ask me anything about Crescent University")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if user_input:
    with st.chat_message("assistant"):
        # Greeting check
        if is_greeting(user_input):
            response = get_greeting_response()
        else:
            norm_input = normalize_input(user_input)
            query_info = parse_query(norm_input)

            # Memory fallback
            if not query_info["department"]:
                last = get_last_query_info()
                if last:
                    query_info["department"] = last.get("department")
                    query_info["faculty"] = last.get("faculty")

            matches = get_courses_for_query(course_data, query_info)
            if matches:
                save_last_query_info(query_info)
                response = "\n\n".join([m["answer"] for m in matches])
            else:
                # Fall back to semantic match
                user_embedding = embedding_model.encode(norm_input)
                scores = faiss_index.search([user_embedding], k=1)[1]
                best_idx = scores[0][0]
                response = qa_data[best_idx]["answer"]
                response_intro = random.choice(["Here’s what I found:", "This might help:", "Check this out:"])
                response = f"{response_intro}\n\n{response}"

        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)
