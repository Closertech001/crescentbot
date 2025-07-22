import streamlit as st
import random
import json
from utils.embedding import embed_query, search_similar_questions
from utils.preprocess import normalize_input
from utils.greetings import detect_greeting, get_random_greeting
from utils.course_query import parse_query, get_courses_for_query
from openai import OpenAI
import time

# 🔐 Load OpenAI API Key
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# 🎨 Page configuration
st.set_page_config(page_title="Crescent University Chatbot", page_icon="🤖")
st.markdown("<style>" + open("assets/style.css").read() + "</style>", unsafe_allow_html=True)

# 📁 Load course and QA data
with open("data/course_data.json") as f:
    course_data = json.load(f)

with open("data/crescent_qa.json") as f:
    crescent_qa = json.load(f)

# 🤖 Dynamic bot replies
BOT_REPLIES = [
    "Here’s what I found for you:",
    "This might help:",
    "I’ve got something for you!",
    "Check this out:",
    "Here you go:",
    "Hope this answers your question:",
    "Look what I found:",
]

# 💬 Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "last_department" not in st.session_state:
    st.session_state.last_department = None

# ⏳ Typing effect
def show_typing():
    with st.empty():
        for dots in [".", "..", "..."]:
            st.markdown(f"🤖 Bot is typing{dots}")
            time.sleep(0.3)

# 🧠 Answer generation
def generate_response(user_input):
    normalized = normalize_input(user_input)
    
    # Greet detection
    if detect_greeting(normalized):
        return get_random_greeting()

    # Try course-related question first
    query_info = parse_query(normalized)
    course_matches = get_courses_for_query(course_data, query_info)

    if course_matches:
        st.session_state.last_department = query_info.get("department")
        responses = [f"**Q:** {m['question']}\n**A:** {m['answer']}" for m in course_matches]
        return f"**{random.choice(BOT_REPLIES)}**\n\n" + "\n\n---\n\n".join(responses)

    # If no course match, try general QA
    embedded_query = embed_query(normalized)
    result = search_similar_questions(embedded_query, crescent_qa, top_k=1)

    if result:
        return f"**{random.choice(BOT_REPLIES)}**\n\n**Q:** {result['question']}\n**A:** {result['answer']}"

    return "🤔 I couldn’t find an answer for that. Try rephrasing your question."

# 💬 Input area
st.title("🎓 Crescent University Chatbot")
user_input = st.text_input("Ask me anything about Crescent University:", key="user_input", label_visibility="collapsed")

# 🗨️ Show messages
for msg in st.session_state.chat_history:
    st.markdown(f"🧑‍🎓 **You:** {msg['user']}")
    st.markdown(f"🤖 **Bot:** {msg['bot']}")
    st.markdown("---")

# 📤 Send & respond
if user_input:
    st.session_state.chat_history.append({"user": user_input, "bot": "..."})
    show_typing()
    response = generate_response(user_input)
    st.session_state.chat_history[-1]["bot"] = response
    st.experimental_rerun()

# 🧹 Sidebar
with st.sidebar:
    if st.button("🧼 Clear Chat"):
        st.session_state.chat_history = []
        st.session_state.user_input = ""
        st.rerun()
