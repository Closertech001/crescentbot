import streamlit as st
import openai
import random
import time

from utils.course_query import get_course_info
from utils.embedding import load_qa_dataset
from utils.greetings import is_greeting, respond_to_greeting
from utils.log_utils import log_interaction
from utils.memory import MemoryHandler
from utils.preprocess import normalize_input
from utils.rewrite import improve_question
from utils.search import find_response
from utils.tone import detect_tone

# Load styling
with open("assets/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Load OpenAI API key
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "memory" not in st.session_state:
    st.session_state.memory = MemoryHandler()

# Load model and embeddings
model, df, embeddings = load_qa_dataset()

# Chat title
st.title("🎓 Crescent University Chatbot")
st.markdown("Ask me anything about departments, courses, or admission.")

# Input box
user_input = st.chat_input("Type your question...")

# Typing animation
def show_typing_indicator():
    message = st.empty()
    for dots in ["", ".", "..", "..."]:
        message.markdown(f"🤖 Bot is typing{dots}")
        time.sleep(0.3)
    message.empty()

# Message rendering
def render_message(speaker, msg):
    bubble_class = "chat-message-user" if speaker == "You" else "chat-message-assistant"
    st.markdown(f'<div class="{bubble_class}"><b>{speaker}:</b><br>{msg}</div>', unsafe_allow_html=True)

# Main interaction
if user_input:
    normalized = normalize_input(user_input)
    tone = detect_tone(normalized)
    st.session_state.chat_history.append(("You", user_input))
    render_message("You", user_input)

    # Bot typing indicator
    show_typing_indicator()

    # Greeting or small talk
    if is_greeting(normalized):
        response = respond_to_greeting()
        render_message("Bot", response)
        st.session_state.chat_history.append(("Bot", response))
        log_interaction(user_input, response, tone)
    else:
        # Try to match course info
        course_answer = get_course_info(normalized, st.session_state.memory)
        if course_answer:
            render_message("Bot", course_answer)
            st.session_state.chat_history.append(("Bot", course_answer))
            log_interaction(user_input, course_answer, tone)
        else:
            # Improve and search
            refined_query = improve_question(normalized, st.session_state.memory)
            answer, related = find_response(refined_query, model, df, embeddings)

            # Dynamic intro phrases
            tone_prefixes = {
                "polite": ["Certainly!", "Of course!", "Glad to help!"],
                "confused": ["Let me clarify that for you.", "Here's what I found."],
                "angry": ["I'm here to assist despite the frustration.", "Let's fix this."],
                "emphatic": ["Here's a strong answer for you!", "Absolutely, here you go:"],
                "urgent": ["Quick answer:", "Right away!"],
                "neutral": ["Here's what I found:", "Let me help with that:"]
            }
            prefix = random.choice(tone_prefixes.get(tone, tone_prefixes["neutral"]))
            full_response = f"{prefix} {answer}"
            render_message("Bot", full_response)
            st.session_state.chat_history.append(("Bot", full_response))
            log_interaction(user_input, full_response, tone)

            # Related questions
            if related:
                st.markdown("#### Related Questions")
                for q in related:
                    st.markdown(f'<div class="related-question">{q}</div>', unsafe_allow_html=True)

# Show chat history
st.divider()
for speaker, msg in st.session_state.chat_history:
    render_message(speaker, msg)

# Clear chat
if st.button("🔄 Clear Chat"):
    st.session_state.chat_history = []
    st.session_state.memory.reset()
    st.experimental_rerun()
