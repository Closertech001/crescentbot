# utils/greetings.py

import random
import re

GREETING_PATTERNS = [
    r"\bhi\b", r"\bhello\b", r"\bhey\b", r"\bgood (morning|afternoon|evening)\b",
    r"\bhow are you\b", r"\bwhat's up\b", r"\bhowdy\b", r"\byo\b",
]

GREETING_RESPONSES = [
    "Hello! 😊 How can I assist you today?",
    "Hi there! 👋 What would you like to know about Crescent University?",
    "Hey! 👋 Feel free to ask me anything about courses or departments.",
    "Welcome! 👨‍🎓 What can I help you with today?",
]

def is_greeting(user_input):
    return any(re.search(pattern, user_input.lower()) for pattern in GREETING_PATTERNS)

def get_greeting_response():
    return random.choice(GREETING_RESPONSES)
