import random
import re

# Sample greetings
GREETINGS = [
    "Hello! How can I assist you today?",
    "Hi there! Ask me anything about Crescent University.",
    "Hey! Ready to help you with courses, departments, or requirements."
]

FAREWELLS = [
    "Goodbye! Feel free to return anytime. 👋",
    "See you later! Wishing you success. 🌟",
    "Take care! I'm here whenever you need me."
]

SMALLTALK_RESPONSES = {
    "who created you": [
        "I was created by a developer using OpenAI and Streamlit for Crescent University. 🤖",
        "I’m powered by AI and built with ❤️ for Crescent University students."
    ],
    "how are you": [
        "I’m doing great, thanks for asking! 😊",
        "I’m functioning perfectly. How can I help you?"
    ],
    "what is your name": [
        "I’m Crescent University’s chatbot. You can call me CresBot! 🎓",
        "Just your friendly campus assistant. 😊"
    ],
    "who are you": [
        "I'm an AI assistant for Crescent University. Here to help you out!",
        "I’m a chatbot designed to guide you through anything Crescent University related."
    ]
}

COURSE_CODE_REGEX = r"\b([A-Z]{2,4}\s?\d{3})\b"


def detect_greeting(text):
    for word in ["hello", "hi", "hey", "good morning", "good afternoon", "good evening"]:
        if word in text.lower():
            return True
    return False

def detect_farewell(text):
    for word in ["bye", "goodbye", "see you", "later", "take care"]:
        if word in text.lower():
            return True
    return False

def get_random_greeting():
    return random.choice(GREETINGS)

def detect_smalltalk(text):
    text = text.lower()
    for phrase in SMALLTALK_RESPONSES:
        if phrase in text:
            return True
    return False

def get_smalltalk_response(text):
    text = text.lower()
    for phrase in SMALLTALK_RESPONSES:
        if phrase in text:
            return random.choice(SMALLTALK_RESPONSES[phrase])
    return "🙂"

def extract_course_code(text):
    match = re.search(COURSE_CODE_REGEX, text.upper())
    if match:
        return match.group(1).replace(" ", "")
    return None

def get_course_by_code(course_code, qa_records):
    course_code = course_code.upper().replace(" ", "")
    for entry in qa_records:
        if course_code in entry.get("question", ""):
            return f"📘 **{entry['question']}**\n\n{entry['answer']}"
    return None
