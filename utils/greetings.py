# utils/greetings.py

import random
import re

# --- GREETING DETECTION ---
GREETING_PATTERNS = [
    r"\bhi\b", r"\bhello\b", r"\bhey\b",
    r"\bgood (morning|afternoon|evening)\b",
    r"\bwhat'?s up\b", r"\bhowdy\b", r"\byo\b",
    r"\bhow are you\b", r"\bhow are things\b"
]

GREETINGS_RESPONSES = [
    "Hey there! 😊 How can I help you today?",
    "Hi! What would you like to know?",
    "Hello! Ready when you are.",
    "Good to see you! How can I assist?",
    "Hi there! Ask me anything.",
]

def is_greeting(text):
    text = text.lower()
    return any(re.search(pattern, text) for pattern in GREETING_PATTERNS)

def get_greeting_response():
    return random.choice(GREETINGS_RESPONSES)

# --- SMALL TALK DETECTION ---
SMALL_TALK_PATTERNS = {
    r"\bhow (are|r) you\b": [
        "I'm doing great! Thanks for asking. 😊",
        "All systems go! 🚀 How can I assist you?",
        "I'm always here and ready to help you out!"
    ],
    r"\bwhat'?s up\b": [
        "Not much, just ready to help you. What’s up with you?",
        "Just hanging out in the cloud ☁️. Need something?",
    ],
    r"\bthank(s| you)\b": [
        "You're welcome! 😄",
        "Anytime!",
        "No problem at all.",
    ],
    r"\b(who are you|what can you do)\b": [
        "I’m the Crescent University helper bot! Ask me about courses, departments, fees, and more.",
        "I assist with info about Crescent University. What would you like to know?",
    ]
}

def is_small_talk(text):
    text = text.lower()
    return any(re.search(pattern, text) for pattern in SMALL_TALK_PATTERNS)

def get_small_talk_response(text):
    text = text.lower()
    for pattern, responses in SMALL_TALK_PATTERNS.items():
        if re.search(pattern, text):
            return random.choice(responses)
    return None
