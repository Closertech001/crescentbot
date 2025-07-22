import streamlit as st
import random
import json
import re
import time
from openai import OpenAI

from utils.embedding import get_top_k_answers
from utils.course_query import parse_query, get_courses_for_query
from utils.greetings import is_greeting, get_greeting_response
from utils.preprocess import normalize_input
from utils.memory import Memory
from utils.memory import store_context_from_query, enrich_query_with_context

# --- Page Setup ---
st.set_page_config(page_title="Crescent University Chatbot")
st.markdown("<style>div[data-testid=\"stSidebar\"]{background-color:#f0f2f6;}</style>", unsafe_allow_html=True)

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# --- Load Data ---
with open("data/crescent_qa.json", "r", encoding="utf-8") as f:
    QA_DATA = [item for item in json.load(f) if "question" in item and "answer" in item]

with open("data/course_data.json", "r", encoding="utf-8") as f:
    COURSE_DATA = json.load(f)

# --- Session Setup ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if "memory" not in st.session_state:
    st.session_state.memory = Memory()

# --- Helpers ---
def bot_typing():
    with st.empty():
        for i in range(3):
            st.markdown("**Bot is typing" + "." * (i + 1) + "**")
            time.sleep(0.4)

def explain_course_level(user_input):
    match = re.search(r"\b(100|200|300|400|500)\s*level\b", user_input.lower())
    if match:
        level = match.group(1)
        year_map = {
            "100": "100 level means Year 1 (Freshman or First Year).",
            "200": "200 level means Year 2 (Sophomore or Second Year).",
            "300": "300 level means Year 3 (Third Year).",
            "400": "400 level means Year 4 (Final Year for most 4-year programs).",
            "500": "500 level means Year 5 (Final Year for programs like Law, Architecture)."
        }
        return year_map.get(level)
    return None

def generate_dynamic_intro():
    return random.choice([
        "Here’s what I found:",
        "Take a look at this:",
        "This might help:",
        "Here’s the answer you’re looking for:",
        "Check this out:",
    ])

# --- UI ---
st.title("🎓 Crescent University Chatbot")

user_input = st.chat_input("Ask me anything about Crescent University")

if user_input:
    normalized_input = normalize_input(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Greet
    if is_greeting(normalized_input):
        response = get_greeting_response()
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()

    # Explain course level
    level_explanation = explain_course_level(normalized_input)
    if level_explanation:
        st.session_state.messages.append({"role": "assistant", "content": level_explanation})
        st.rerun()

    bot_typing()

    # Try course-based response
    query_info = parse_query(normalized_input)
    matched_courses = get_courses_for_query(query_info, COURSE_DATA)

    if matched_courses:
        st.session_state.memory.update(query_info)
        intro = generate_dynamic_intro()
        course_responses = [f"- **{m['question']}**\n{m['answer']}" for m in matched_courses]
        response = intro + "\n" + "\n\n".join(course_responses)
    else:
        # Use memory to enhance fallback query
        enriched_input = normalized_input
        mem = st.session_state.memory.get_memory()
        if mem.get("departments"):
            enriched_input += " for department " + ", ".join(mem["departments"])
        if mem.get("level"):
            enriched_input += f" {mem['level']} level"
        if mem.get("semester"):
            enriched_input += f" {mem['semester']} semester"

        try:
            results = get_top_k_answers(enriched_input, top_k=3)
            if results:
                intro = generate_dynamic_intro()
                response = intro + "\n" + "\n\n".join([
                    f"**Q:** {q}\n**A:** {a}" for (a, score) in results for q in [q for q, s in results if s == score]
                ])
            else:
                raise ValueError("No top results")
        except Exception as e:
            try:
                gpt_response = client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant for Crescent University."},
                        {"role": "user", "content": user_input}
                    ]
                )
                response = gpt_response.choices[0].message.content
            except Exception as e:
                response = "Sorry, I'm having trouble answering right now. Please try again later."

    st.session_state.messages.append({"role": "assistant", "content": response})

# Display messages
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
