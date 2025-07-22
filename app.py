import streamlit as st
import json
import time
from utils.embedding import load_model, load_dataset, get_question_embeddings
from utils.course_query import parse_query, get_courses_for_query
from utils.greetings import is_greeting, get_greeting_response
from utils.preprocess import normalize_input
from utils.memory import store_context_from_query, enrich_query_with_context
from openai import OpenAI
import random

st.set_page_config(page_title="Crescent University Chatbot")
st.markdown("<style>div[data-testid=\"stSidebar\"]{background-color:#f0f2f6;}</style>", unsafe_allow_html=True)

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Load data once
QA_DATA, questions = load_dataset("data/crescent_qa.json")
COURSE_DATA = None
with open("data/course_data.json", "r", encoding="utf-8") as f:
    COURSE_DATA = json.load(f)

# Load model and embeddings once
model = load_model()
embeddings = get_question_embeddings(questions, model)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "memory" not in st.session_state:
    st.session_state.memory = {}

def bot_typing():
    with st.empty():
        for i in range(3):
            st.markdown("**Bot is typing" + "." * (i + 1) + "**")
            time.sleep(0.4)

def explain_course_level(user_input):
    import re
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
    choices = [
        "Here’s what I found:",
        "Take a look at this:",
        "This might help:",
        "Here’s the answer you’re looking for:",
        "Check this out:",
    ]
    return random.choice(choices)

st.title("🎓 Crescent University Chatbot")

user_input = st.chat_input("Ask me anything about Crescent University")

if user_input:
    normalized_input = normalize_input(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Handle greeting
    if is_greeting(normalized_input):
        response = get_greeting_response()
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.experimental_rerun()

    # Handle level explanation
    level_explanation = explain_course_level(normalized_input)
    if level_explanation:
        st.session_state.messages.append({"role": "assistant", "content": level_explanation})
        st.experimental_rerun()

    bot_typing()

    # Try course query first
    query_info = parse_query(normalized_input)
    matched_courses = get_courses_for_query(query_info, COURSE_DATA)

    if matched_courses:
        st.session_state.memory.update(query_info)
        intro = generate_dynamic_intro()
        course_responses = [f"- **{m['question']}**\n{m['answer']}" for m in matched_courses]
        response = intro + "\n\n" + "\n\n".join(course_responses)
    else:
        # Use last memory to improve fallback
        enriched_input = normalized_input
        mem = st.session_state.memory
        if mem.get("departments"):
            enriched_input += " for department " + ", ".join(mem["departments"])
        if mem.get("level"):
            enriched_input += f" {mem['level']} level"
        if mem.get("semester"):
            enriched_input += f" {mem['semester']} semester"

        # Semantic search fallback on QA dataset
        # Compute similarity with precomputed embeddings
        query_embedding = model.encode([enriched_input], convert_to_numpy=True, normalize_embeddings=True)
        import faiss
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)
        D, I = index.search(query_embedding, 3)  # top 3
        top_indices = I[0]
        top_scores = D[0]

        # Prepare response from top matches
        matches = []
        for idx, score in zip(top_indices, top_scores):
            if score > 0.6:
                entry = QA_DATA[idx]
                matches.append((entry["question"], entry["answer"], score))

        if matches:
            intro = generate_dynamic_intro()
            response = intro + "\n\n"
            for q, a, s in matches:
                response += f"**Q:** {q}\n**A:** {a}\n\n"
        else:
            # fallback to GPT if no confident match
            gpt_response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant for Crescent University. Answer questions clearly and accurately."},
                    {"role": "user", "content": user_input}
                ]
            )
            response = gpt_response.choices[0].message.content

    st.session_state.messages.append({"role": "assistant", "content": response})

# Display messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
