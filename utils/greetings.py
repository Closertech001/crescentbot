import random
import re

# --- Greeting Keywords ---
GREETING_KEYWORDS = [
    "hello", "hi", "hey", "yo", "howdy", "hiya", "sup", "what's up",
    "good morning", "good afternoon", "good evening", "greetings"
]

GREETING_RESPONSES = [
    "Hello! 👋 How can I assist you with Crescent University today?",
    "Hi there! 😊 What would you like to know about Crescent University?",
    "Hey! 👋 Feel free to ask me anything related to Crescent University.",
    "Greetings! 🤝 I’m here to help with your Crescent University questions.",
    "Yo! 👋 Ready to chat about Crescent University?",
    "Hiya! 🙋 Let me know what you need help with at Crescent University.",
    "Howdy! 🤠 I'm your friendly assistant for all things Crescent University.",
    "Good to see you! 🌟 Got any questions about Crescent University?"
]

# --- Small Talk Keywords ---
SMALL_TALK_PATTERNS = {
    r"\bhow are you\b": [
        "I'm doing great, thank you! 😊 How can I help you today?",
        "All systems go! Ready to assist you with Crescent University questions. 🚀",
        "Feeling helpful as always! What can I do for you today? 💡"
    ],
    r"\bwhat(?:'s| is) your name\b": [
        "You can call me CrescentBot 🤖, your helpful university assistant.",
        "I'm CrescentBot! I specialize in answering all your Crescent University questions. 🎓"
    ],
    r"\btell me about yourself\b": [
        "I'm CrescentBot, built to assist students and staff with everything about Crescent University! 💬",
        "I’m a smart assistant trained on Crescent University data to help with inquiries, courses, departments and more. 🧠"
    ],
    r"\bwho (are|r) you\b": [
        "I'm CrescentBot, your AI guide to Crescent University. 🎓 Ask me anything!",
        "I’m your virtual assistant, here to help with all things Crescent University!"
    ],
    r"\bthank(s| you)\b": [
        "You're very welcome! 😊",
        "Anytime! Let me know if you have more questions. 🙌",
        "Glad to help! 👍"
    ],
    r"\bgood (job|bot|work)\b": [
        "Thank you! 🤖 I’m here to help anytime!",
        "I appreciate that! 😊",
        "Yay! I'm glad I could help. 💙"
    ]
}

# --- Greeting Detection ---
def is_greeting(text):
    text = text.lower()
    return any(re.search(rf"\b{kw}\b", text) for kw in GREETING_KEYWORDS)

# --- Small Talk Detection ---
def is_small_talk(text):
    text = text.lower()
    for pattern in SMALL_TALK_PATTERNS:
        if re.search(pattern, text):
            return True
    return False

# --- Generate Greeting Response ---
def greeting_responses(text=None):
    return random.choice(GREETING_RESPONSES)

# --- Generate Small Talk Response ---
def small_talk_response(text):
    text = text.lower()
    for pattern, responses in SMALL_TALK_PATTERNS.items():
        if re.search(pattern, text):
            return random.choice(responses)
    return None
