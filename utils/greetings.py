import re
import random

# 🎉 Greeting detection
GREETING_KEYWORDS = ["hello", "hi", "hey", "greetings", "good morning", "good afternoon", "good evening"]

GREETING_RESPONSES = [
    "Hello! 👋 What would you like to know about Crescent University?",
    "Hi there! 😊 Ask me anything about courses, departments, or school info.",
    "Greetings! 🎓 How can I help you today?",
    "Hey! Feel free to ask me anything about Crescent University. 📘",
    "Welcome! I'm here to assist you with Crescent University queries. 🙌"
]

# 💬 Small Talk patterns
SMALL_TALK_PATTERNS = {
    r"\bhow are you\b": [
        "I'm doing great! 😊 How about you?",
        "All systems go! ⚙️ What can I help you with today?",
        "Feeling chatty and ready to assist! 💬"
    ],
    r"\bwhat can you do\b": [
        "I can help you with information about Crescent University – courses, departments, admissions, and more! 🎓",
        "Ask me about course codes, fees, faculty, departments, and everything about Crescent! 🏫"
    ],
    r"\bwho (are|r) you\b": [
        "I’m your friendly Crescent University assistant chatbot 🤖",
        "I'm here to guide you with info about Crescent University. 😊"
    ],
    r"\bthank(s| you)\b": [
        "You're welcome! 😊",
        "Anytime! Let me know if you need more help. 🙌",
        "Glad I could help! 🤗"
    ],
    r"\bgoodbye\b|\bbye\b": [
        "Goodbye! 👋 Come back anytime you need help.",
        "See you later! All the best at Crescent University! 🎓"
    ]
}

# ✅ Greeting matcher
def is_greeting(text):
    text = text.lower()
    return any(kw in text for kw in GREETING_KEYWORDS)

def greeting_responses(text=None):
    return random.choice(GREETING_RESPONSES)

# ✅ Small Talk matcher
def is_small_talk(text):
    text = text.lower()
    return any(re.search(pattern, text) for pattern in SMALL_TALK_PATTERNS)

def small_talk_response(text):
    text = text.lower()
    for pattern, responses in SMALL_TALK_PATTERNS.items():
        if re.search(pattern, text):
            return random.choice(responses)
    return "Interesting! 😊 Let’s get back to university-related stuff."

# 🔍 Course code extractor
def extract_course_code(text):
    match = re.search(r"\b([A-Z]{2,4})[-\s]?(CSC|GST|PHY|MTH|BIO|CHE|CUAB|CSC)?[-\s]?(\d{3})\b", text.upper())
    if match:
        return "".join([part for part in match.groups() if part])
    return None

# 📘 Get course details from data
def get_course_by_code(code, course_data):
    code = code.upper()
    for entry in course_data:
        if code in entry.get("question", "").upper():
            return entry["answer"]
    return None
