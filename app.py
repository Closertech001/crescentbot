import streamlit as st
import json
import time

from utils.embedding import load_model, load_dataset, compute_question_embeddings
from utils.course_query import parse_query, get_courses_for_query
from utils.greetings import is_greeting, get_greeting_response
from utils.preprocess import normalize_input
from utils.memory import store_context_from_query, enrich_query_with_context
from utils.rewrite import rewrite_with_tone
from openai import OpenAI

st.set_page_config(page_title="Crescent University Chatbot", layout="wide")
st.markdown("<style>div[data-testid=\"stSidebar\"]{background-color:#f0f2f6;}</style>", unsafe_allow_html=True)

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# --- Cache resources so we don't reload each interaction ---
@st.cache_resource
def load_resources():
    model = load_model()
    qa_df = load_dataset("data/crescent_qa.json")
    questions = qa_df["question"].tolist()
    embeddings = compute_question_embeddings(questions, model)
    with open("data/course_data.json", "r", encoding="utf-8") as f:
        course_data = json.load(f)
    return model, qa_df, embeddings, course_data

model, qa_df, embeddings, COURSE_DATA = load_resources()

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []

if "memory" not in st.session_state:
    # Simple dictionary to hold last department/level etc.
    st.session_state.memory = {}

def bot_typing():
    with st.empty():
        for i in range(3):
            st.markdown("**Bot is typing" + "." * (i + 1) + "**")
            time.sleep(0.3)

def generate_dynamic_intro():
    intros = [
        "Here's what I found for you:",
        "Check this out:",
        "Take a look:",
        "This might help:",
        "Here's the information you requested:"
    ]
    import random
    return random.choice(intros)

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

# Main Streamlit app
st.title("🎓 Crescent University Chatbot")

user_input = st.chat_input("Ask me anything about Crescent University")

if user_input:
    normalized_input = normalize_input(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Handle greeting
    if is_greeting(normalized_input):
        response = get_greeting_response()

    else:
        # Explain course level if asked
        level_explanation = explain_course_level(normalized_input)
        if level_explanation:
            response = level_explanation
        else:
            bot_typing()

            # Parse query and enrich with memory context
            query_info = parse_query(normalized_input)
            query_info = enrich_query_with_context(query_info)

            matched_courses = get_courses_for_query(query_info, COURSE_DATA)

            if matched_courses:
                # Update memory for next queries
                st.session_state.memory.update(query_info)
                intro = generate_dynamic_intro()
                course_texts = [f"- **{entry['question']}**\n{entry['answer']}" for entry in matched_courses]
                response = intro + "\n\n" + "\n\n".join(course_texts)

            else:
                # Semantic search fallback on QA dataset
                from utils.embedding import get_top_k_answers

                # Updated get_top_k_answers to accept model, dataset, embeddings
                top_answers = get_top_k_answers(normalized_input, model=model, dataset=qa_df, embeddings=embeddings, top_k=3)

                if top_answers:
                    intro = generate_dynamic_intro()
                    response = intro + "\n\n" + "\n\n".join([f"**A:** {ans}" for ans, score in top_answers])
                else:
                    # Final fallback to GPT
                    gpt_resp = client.chat.completions.create(
                        model="gpt-4",
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant for Crescent University. Answer clearly and politely."},
                            {"role": "user", "content": user_input}
                        ]
                    )
                    response = gpt_resp.choices[0].message.content

    # Optionally rewrite response based on detected tone
    response = rewrite_with_tone(user_input, response)

    st.session_state.messages.append({"role": "assistant", "content": response})

# Display chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
