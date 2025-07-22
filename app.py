import streamlit as st
import openai
from utils.embedding import load_model_and_embeddings, find_most_similar_question
from utils.course_query import extract_course_info
from utils.preprocess import normalize_input
from utils.memory import MemoryHandler
from utils.greetings import detect_greeting, generate_greeting_response, is_small_talk, generate_small_talk_response
from utils.rewrite import rewrite_followup
from utils.search import fallback_to_gpt_if_needed

# 🔐 OpenAI key setup
try:
    openai.api_key = st.secrets["OPENAI_API_KEY"]
except KeyError:
    st.error("❌ OpenAI API key not found. Please set OPENAI_API_KEY in .streamlit/secrets.toml")
    st.stop()

# 🚀 Load model and FAISS index (cached)
model, qa_data, index, embeddings = load_model_and_embeddings("data/crescent_qa.json")

# 🧠 Memory for conversation state
memory = MemoryHandler()

# 🔍 Confidence threshold for match
CONFIDENCE_THRESHOLD = 0.75

# 💬 Main chatbot function
def crescent_chatbot():
    st.set_page_config(page_title="🎓 Crescent University Chatbot", layout="centered")
    st.markdown("<h1 style='text-align: center;'>🎓 Crescent University Chatbot 🤖</h1>", unsafe_allow_html=True)

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    user_input = st.chat_input("Ask me anything about Crescent University...")

    if user_input and user_input.strip():
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("🔎 Let me check that for you..."):
                response = get_bot_response(user_input)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

# 🤖 Bot logic
def get_bot_response(user_input):
    normalized_input = normalize_input(user_input)

    # Only respond to greetings if that's *all* the user said
    if detect_greeting(normalized_input) and len(normalized_input.split()) <= 3:
        return generate_greeting_response()

    if is_small_talk(normalized_input):
        return generate_small_talk_response()

    # Rewrite follow-up queries using memory
    user_input = rewrite_followup(user_input, memory)

    # Try course-related question
    course_response = extract_course_info(user_input, memory)
    if course_response:
        return course_response

    # Semantic search via FAISS
    best_match, score = find_most_similar_question(user_input, model, index, qa_data)
    if score > CONFIDENCE_THRESHOLD:
        memory.update_last_topic(best_match)
        return best_match["answer"]

    # Fallback to GPT
    return fallback_to_gpt_if_needed(user_input)

# 🔧 Run app
if __name__ == "__main__":
    crescent_chatbot()
