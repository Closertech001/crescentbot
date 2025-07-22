import streamlit as st
import json
import random
import time
from utils.embedding import load_embeddings, search_similar
from utils.course_query import parse_query, get_courses_for_query
from utils.preprocess import normalize_input
from utils.greetings import is_greeting, get_greeting_response

# 🔄 Load data
with open("data/course_data.json", "r", encoding="utf-8") as f:
    course_data = json.load(f)

with open("data/crescent_qa.json", "r", encoding="utf-8") as f:
    qa_data = json.load(f)

embeddings, questions = load_embeddings(qa_data)

# 🎨 Page config
st.set_page_config(page_title="Crescent University Chatbot", page_icon="🎓")
st.markdown('<style>' + open('assets/style.css').read() + '</style>', unsafe_allow_html=True)

# 💬 Chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# 🧠 Small talk responses
RESPONSES = [
    "Here’s what I found for you:",
    "Hope this helps:",
    "Got it! Here you go:",
    "This should answer your question:",
    "Alright, take a look at this:"
]

# ⏳ Typing animation
def bot_typing():
    with st.empty():
        for dots in ["", ".", "..", "..."]:
            st.markdown(f"**Bot is typing{dots}**")
            time.sleep(0.3)

# 🧾 Main interface
st.title("🤖 Crescent University Chatbot")

user_input = st.text_input("Ask me anything about Crescent University...", key="user_input")

if user_input:
    st.session_state.chat_history.append(("user", user_input))
    normalized = normalize_input(user_input)

    # 👋 Greeting
    if is_greeting(normalized):
        response = get_greeting_response()
    else:
        # 📚 Course code handling
        match = None
        course_code_pattern = r"\b[A-Z]{2,4}\s?\d{3}\b"
        code_match = re.search(course_code_pattern, user_input, re.IGNORECASE)

        if code_match:
            course_code = code_match.group().replace(" ", "").upper()
            for entry in course_data:
                if entry.get("course_code", "").replace(" ", "").upper() == course_code:
                    title = entry.get("course_title", "Unknown title")
                    unit = entry.get("course_unit", "N/A")
                    response = f"""📘 *Here’s the info for* `{course_code}`:\n\n{title} ({unit} unit{'s' if unit != 1 else ''})"""
                    match = True
                    break

        # 🔎 Deep query
        if not match:
            query_info = parse_query(normalized)
            if query_info["department"]:
                matched_courses = get_courses_for_query(course_data, query_info)
                if matched_courses:
                    response = f"📚 Courses for **{query_info['department'].title()}**"
                    if query_info["level"]:
                        response += f", Level {query_info['level']}"
                    if query_info["semester"]:
                        response += f", {query_info['semester']} Semester"
                    response += ":\n\n"
                    for course in matched_courses:
                        response += f"- `{course.get('course_code', 'N/A')}`: {course.get('course_title', 'N/A')} ({course.get('course_unit', 'N/A')} units)\n"
                else:
                    response = "I couldn't find courses matching that info. Please try specifying the department or level more clearly."
            else:
                # 🤖 Semantic Q&A fallback
                top_match = search_similar(normalized, embeddings, questions, qa_data)
                if top_match:
                    response = f"{random.choice(RESPONSES)}\n\n{top_match['answer']}"
                else:
                    response = "Sorry, I couldn’t find an answer for that."

    # 🤖 Show bot typing
    bot_typing()
    st.session_state.chat_history.append(("bot", response))
    st.experimental_rerun()

# 📜 Display chat
for sender, msg in st.session_state.chat_history:
    if sender == "user":
        st.markdown(f"**You:** {msg}")
    else:
        st.markdown(f"**Bot:** {msg}")
