import streamlit as st

# Must be the first Streamlit command
st.set_page_config(page_title="🎓 Crescent University Chatbot", layout="centered")

import json
import time
import random
from utils.memory import MemoryHandler
from utils.embedding import load_model_and_embeddings
from utils.course_query import extract_course_info
from utils.greetings import detect_greeting, generate_greeting_response
from utils.rewrite import rewrite_followup
from utils.search import fallback_to_gpt_if_needed

# 🔐 OpenAI key setup
try:
    from openai import OpenAI
    import openai
    openai.api_key = st.secrets["OPENAI_API_KEY"]
except Exception:
    st.warning("🔑 Please set your OpenAI API key in Streamlit Secrets.", icon="⚠️")
    st.stop()

# 🎯 Load data + embeddings
model, qa_data, faiss_index, all_embeddings = load_model_and_embeddings()

# 🧠 Memory handler
memory = MemoryHandler()

# 🎨 App UI
st.title("🎓 Crescent University Chatbot")
st.caption("Ask me anything about Crescent University: courses, units, departments, requirements and more.")

# 🗨️ User input
user_input = st.text_input("💬 Your question:", placeholder="e.g., What is the unit of GST 101 in 100 level?")

if user_input:
    with st.spinner("🤖 Let me check that for you..."):
        question = user_input.strip().lower()

        # 1. Greet user
        if detect_greeting(question):
            st.success(generate_greeting_response())
        else:
            # 2. Try extracting course info
            course_answer = extract_course_info(question, memory)

            if course_answer:
                st.success(course_answer)
            else:
                # 3. Try embedding search
                from sentence_transformers import util
                user_embedding = model.encode(question, convert_to_numpy=True)
                scores, indices = faiss_index.search(user_embedding.reshape(1, -1), k=1)

                best_score = scores[0][0]
                best_index = indices[0][0]

                threshold = 0.35
                if best_score < threshold:
                    # 4. Use GPT fallback
                    rewritten = rewrite_followup(question)
                    gpt_response = fallback_to_gpt_if_needed(rewritten, qa_data)
                    st.info(gpt_response)
                else:
                    matched_answer = qa_data[best_index]["answer"]
                    st.success(matched_answer)

# 💡 Footer
st.markdown("---")
st.caption("Built with ❤️ for Crescent University by thywillmartins")
