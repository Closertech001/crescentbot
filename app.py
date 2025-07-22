import streamlit as st
import random
from utils.embedding import load_model, load_dataset, compute_question_embeddings
from utils.course_query import get_course_info
from utils.greetings import is_greeting, get_random_greeting, is_farewell
from utils.preprocess import normalize_input
from utils.memory import update_last_context, get_last_context
from utils.search import search_similar_qa

# Load model and data
model = load_model()
df = load_dataset("data/crescent_qa.json")
questions = df["question"].tolist()
embeddings = compute_question_embeddings(questions, model)

# App UI
st.set_page_config(page_title="Crescent University Chatbot", layout="wide")
st.markdown("<h2 style='text-align: center;'>🤖 Crescent University Chatbot</h2>", unsafe_allow_html=True)

# Session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_department" not in st.session_state:
    st.session_state.last_department = None
if "last_level" not in st.session_state:
    st.session_state.last_level = None

# Typing animation
def typing_animation():
    with st.empty():
        for dots in ["", ".", "..", "..."]:
            st.markdown(f"<p>🤖 Bot is typing{dots}</p>", unsafe_allow_html=True)
            st.sleep(0.3)

# Display chat history
for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        st.markdown(f"<div style='text-align: right;'>🧑‍💬 **You:** {msg['content']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div style='text-align: left;'>🤖 **Bot:** {msg['content']}</div>", unsafe_allow_html=True)

# User input
user_input = st.text_input("Ask me anything about Crescent University:", key="user_input")
submit = st.button("Send")

if submit and user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    user_message = normalize_input(user_input)

    # Respond to greetings or farewell
    if is_greeting(user_message):
        bot_response = get_random_greeting()
    elif is_farewell(user_message):
        bot_response = random.choice(["Goodbye!", "Take care!", "See you later!", "Farewell!"])
    else:
        # Check if it's a course code query
        course_info = get_course_info(user_message)
        if course_info:
            course_code, course_title, course_unit = course_info
            bot_response = f"""📘 *Here’s the info for* `{course_code}`:

{course_title} ({course_unit} unit{'s' if course_unit != 1 else ''})"""
        else:
            # Try to retrieve context (department/level) if needed
            user_message, memory_update = get_last_context(user_message, st.session_state)
            if memory_update:
                update_last_context(user_message, st.session_state)

            # Search for best matching Q&A
            top_result = search_similar_qa(user_message, questions, embeddings, model, df)

            if top_result:
                response_prefixes = [
                    "Here’s what I found for you:",
                    "This might help:",
                    "Got it! Here’s the answer:",
                    "Let me assist you with this:",
                    "Check this out:"
                ]
                bot_response = f"💡 {random.choice(response_prefixes)}\n\n{top_result}"
            else:
                bot_response = "❌ Sorry, I couldn't find a relevant answer. Could you rephrase your question?"

    # Simulate typing
    typing_animation()

    # Show bot response and update history
    st.markdown(f"<div style='text-align: left;'>🤖 **Bot:** {bot_response}</div>", unsafe_allow_html=True)
    st.session_state.chat_history.append({"role": "bot", "content": bot_response})

    # Clear input
    st.session_state.user_input = ""
