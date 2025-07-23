import re
import random

# --- GREETING DETECTION ---

GREETING_PATTERNS = [
    r"\bhi\b", r"\bhello\b", r"\bhey\b", r"\bgood (morning|afternoon|evening)\b",
    r"\bwhat's up\b", r"\bhowdy\b", r"\byo\b"
]

GREETING_RESPONSES = [
    "Hello! 👋 How can I help you today?",
    "Hi there! 😊 What would you like to know about Crescent University?",
    "Hey! 👋 Ask me anything about courses, departments, or admission.",
    "Good to see you! 👋 What can I do for you today?",
]

def detect_greeting(text):
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in GREETING_PATTERNS)


# --- SMALL TALK DETECTION ---

SMALL_TALK_PATTERNS = [
    r"\bhow are you\b", r"\bwhat's up\b", r"\bwhat are you doing\b", r"\bhow's it going\b",
    r"\bare you (okay|fine)\b", r"\bhow do you do\b"
]

SMALL_TALK_RESPONSES = [
    "I'm doing great, thanks for asking! 😊 How can I assist you today?",
    "All good here! Let me know what you need help with.",
    "I'm here to help with anything about Crescent University!",
    "Feeling smart and ready to assist! 😄"
]

def detect_small_talk(text):
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in SMALL_TALK_PATTERNS)

def respond_to_small_talk(text):
    return random.choice(SMALL_TALK_RESPONSES)
