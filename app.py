import streamlit as st
import os
import time
from dotenv import load_dotenv
from utils.embedding import load_model, load_dataset, compute_question_embeddings
from utils.memory import init_memory
from utils.search import find_response
from utils.log_utils import log_query

# Load environment variables (for OpenAI API key)
load_dotenv()
import openai
openai.api_key = os.getenv("OPENAI_API_KEY")

# Apply custom CSS style
with open("assets/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# App title and intro
st.title("🎓 Crescent University Chatbot")
st.markdown("Ask me anything about courses, departments, or university-related questions.")

# Initialize session memory
init_memory()

# Load model, dataset, and precompute embeddings
@st.cache_resource
def get_bot_resources():
    model = load_model()
    dataset = load_dataset()
    embeddings = compute_question_embeddings(dataset["question"].tolist(), model)
    return model, dataset, embeddings

model, dataset, embeddings = get_bot_resources()

# User input
user_input = st.text_input("💬 Type your question here:")

if user_input:
    with st.spinner("Thinking..."):
        response, department, score, related = find_response(user_input, dataset, embeddings)

        # Save log
        log_query(user_input, score)

        # Update session memory
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        # Typing animation for assistant response
        placeholder = st.empty()
        animated_response = ""
        for word in response.split():
            animated_response += word + " "
            placeholder.markdown(f'<div class="chat-message-assistant">{animated_response.strip()}</div>', unsafe_allow_html=True)
            time.sleep(0.05)
        st.session_state.chat_history.append({"role": "assistant", "content": response})

        st.session_state.related_questions = related
        st.session_state.last_department = department

# Display chat history
for msg in st.session_state.chat_history:
    role_class = "chat-message-user" if msg["role"] == "user" else "chat-message-assistant"
    st.markdown(f'<div class="{role_class}">{msg["content"]}</div>', unsafe_allow_html=True)

# Show department label if detected
if st.session_state.last_department:
    st.markdown(f'<div class="department-label">Department: {st.session_state.last_department}</div>', unsafe_allow_html=True)

# Show related questions
if st.session_state.related_questions:
    st.markdown("**🔁 Related Questions:**")
    for q in st.session_state.related_questions:
        if st.button(q):
            st.session_state.chat_history.append({"role": "user", "content": q})
            with st.spinner("Thinking..."):
                response, department, score, related = find_response(q, dataset, embeddings)

                # Typing animation for assistant response to related question
                placeholder = st.empty()
                animated_response = ""
                for word in response.split():
                    animated_response += word + " "
                    placeholder.markdown(f'<div class="chat-message-assistant">{animated_response.strip()}</div>', unsafe_allow_html=True)
                    time.sleep(0.05)
                st.session_state.chat_history.append({"role": "assistant", "content": response})

                st.session_state.related_questions = related
                st.session_state.last_department = department
                st.experimental_rerun()
