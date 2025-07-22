import streamlit as st
from utils.course_query import parse_query, get_courses_for_query
from utils.embedding import load_model, load_dataset, compute_question_embeddings
from utils.search import search_similar
from utils.preprocess import normalize_input
import random
import openai
import time

# 🌐 Page config
st.set_page_config(page_title="Crescent University Chatbot", page_icon="🎓", layout="centered")
st.markdown('<style>' + open("assets/style.css").read() + '</style>', unsafe_allow_html=True)

# 🔐 OpenAI Key
openai.api_key = st.secrets["OPENAI_API_KEY"]

# ✅ Inlined greetings logic
def detect_greeting(text):
    greetings = ["hi", "hello", "hey", "greetings", "good morning", "good afternoon", "good evening"]
    return any(word in text.lower() for word in greetings)

def detect_farewell(text):
    farewells = ["bye", "goodbye", "see you", "take care", "later", "ciao", "farewell"]
    return any(word in text.lower() for word in farewells)

def get_random_greeting():
    return random.choice([
        "Hello! 👋 How can I help you today?",
        "Hi there! 😊 Ask me anything about Crescent University.",
        "Hey! Ready to assist you with your questions.",
        "Greetings! What would you like to know?",
        "Hi! Need help with anything from CUAB?"
    ])

# 📦 Load everything once
@st.cache_resource
def setup():
    model = load_model()
    df = load_dataset()
    embeddings = compute_question_embeddings(df['question'].tolist(), model)
    return model, df, embeddings

model, qa_df, qa_embeddings = setup()

# 🧠 Init session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_department" not in st.session_state:
    st.session_state.last_department = None
if "last_level" not in st.session_state:
    st.session_state.last_level = None
if "last_topic" not in st.session_state:
    st.session_state.last_topic = None

# 💬 Typing animation
def bot_typing_effect():
    with st.empty():
        for dots in ["", ".", "..", "..."]:
            st.markdown(f"**Bot is typing{dots}**")
            time.sleep(0.3)

# 🤖 Chat handler
def handle_input(user_input):
    normalized = normalize_input(user_input)

    # Greeting/Farewell
    if detect_greeting(normalized):
        return get_random_greeting()
    if detect_farewell(normalized):
        return "Goodbye! Feel free to return anytime. 👋"

    # Extract query info
    query_info = parse_query(normalized)

    # Use memory fallback
    if not query_info.get("department") and st.session_state.last_department:
        query_info["department"] = st.session_state.last_department
    if not query_info.get("level") and st.session_state.last_level:
        query_info["level"] = st.session_state.last_level

    # Update memory
    if query_info.get("department"):
        st.session_state.last_department = query_info["department"]
    if query_info.get("level"):
        st.session_state.last_level = query_info["level"]

    # 🎯 Check if it's a course-related query
    course_results = get_courses_for_query(query_info, qa_df.to_dict(orient="records"))
    if course_results:
        response = "📚 **Here’s what I found:**\n\n"
        for r in course_results:
            response += f"- **{r['question']}**\n    {r['answer']}\n\n"
        return response.strip()

    # 🔍 Try semantic search
    top_result = search_similar(normalized, qa_df, qa_embeddings, model)
    if top_result and top_result['score'] > 0.75:
        st.session_state.last_topic = top_result["question"]
        return f"💡 {random.choice(['Here you go:', 'I found this for you:', 'This might help:', 'Check this out:'])}\n\n{top_result['answer']}"

    # 🤖 Fallback to GPT
    try:
        bot_typing_effect()
        completion = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant for Crescent University answering admission, departmental, and course-related questions."},
                {"role": "user", "content": user_input}
            ]
        )
        return completion.choices[0].message.content.strip()
    except Exception:
        return "⚠️ I’m having trouble fetching that. Please try again later."

# 🧑‍💻 UI
st.title("🎓 Crescent University Chatbot")
user_input = st.text_input("Ask me anything about the university...", key="user_input")

if user_input:
    response = handle_input(user_input)
    st.session_state.chat_history.append(("You", user_input))
    st.session_state.chat_history.append(("Bot", response))
    st.session_state.user_input = ""

# 📝 Chat history
for sender, msg in st.session_state.chat_history:
    st.markdown(f"**{sender}:** {msg}")
