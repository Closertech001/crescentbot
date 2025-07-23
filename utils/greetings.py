import re
import random

# 🎉 Greeting keywords
GREETING_KEYWORDS = [
    "hi", "hello", "hey", "greetings", "good morning",
    "good afternoon", "good evening", "what's up", "howdy"
]

GREETING_RESPONSES = [
    "Hello! 👋 What would you like to know about Crescent University?",
    "Hi there! 😊 Feel free to ask me anything.",
    "Greetings! How can I assist you today?",
    "Hey! I’m here to help with any Crescent-related question.",
    "Welcome! Ask me anything about your department, courses, or requirements."
]

# 💬 Small talk triggers and responses
SMALL_TALK = {
    "how are you": [
        "I'm great, thanks for asking! 😊",
        "Doing well and ready to help! What can I do for you?"
    ],
    "who are you": [
        "I’m the Crescent University Chatbot, here to help you with academic info!",
        "I'm a virtual assistant created to help Crescent students with any question."
    ],
    "who made you": [
        "I was built by a student as a project using AI tools! 🤖",
        "I’m powered by AI, built as part of a university chatbot project."
    ],
    "thank you": [
        "You're welcome! 😊",
        "Anytime! Let me know if you need anything else.",
        "Glad I could help!"
    ],
    "thanks": [
        "You're welcome!",
        "Happy to help! 👍"
    ],
    "ok": [
        "Alright! Let me know if there’s anything else.",
        "Sure, I’m here when you need me."
    ],
    "bye": [
        "Goodbye! 👋",
        "See you later!",
        "Take care!"
    ]
}

def is_greeting(text):
    text = text.lower()
    return any(word in text for word in GREETING_KEYWORDS)

def greeting_responses(text=None):
    return random.choice(GREETING_RESPONSES)

def is_small_talk(text):
    text = text.lower()
    return any(trigger in text for trigger in SMALL_TALK)

def small_talk_response(text):
    text = text.lower()
    for trigger, responses in SMALL_TALK.items():
        if trigger in text:
            return random.choice(responses)
    return "I'm not sure how to respond to that yet 😅"

# 🔍 Extract course code like "CSC 101" or "MTH101"
COURSE_CODE_PATTERN = re.compile(r"\b([A-Z]{2,4})[-\s]?(\d{3})\b", re.IGNORECASE)

def extract_course_code(text):
    match = COURSE_CODE_PATTERN.search(text)
    if match:
        return f"{match.group(1).upper()} {match.group(2)}"
    return None

# 📘 Lookup course in course_data
def get_course_by_code(code, course_data):
    code = code.replace("-", " ").upper()
    for entry in course_data:
        if code in entry.get("question", "").upper():
            return entry["answer"]
    return None
