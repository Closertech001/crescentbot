# utils/greetings.py

import random
import re

# --- GREETING DETECTION ---

GREETING_PATTERNS = [
    r"\bhi\b",
    r"\bhello\b",
    r"\bhey\b",
    r"\bgood (morning|afternoon|evening)\b",
    r"\bwhat's up\b",
    r"\bhowdy\b",
    r"\byo\b",
    r"\bsup\b"
]

def is_greeting(text):
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in GREETING_PATTERNS)

def get_greeting_response():
    responses = [
        "Hello! How can I assist you today?",
        "Hi there! What would you like to know about Crescent University?",
        "Hey! Need help with something related to Crescent University?",
        "Greetings! I'm here to assist you with any CUAB info you need."
    ]
    return random.choice(responses)

# --- SMALL TALK DETECTION ---

SMALL_TALK_PATTERNS = {
    r"\bhow are you\b": [
        "I'm doing great, thanks for asking! How can I help you today?",
        "I'm well! Ready to help you with your Crescent University queries.",
        "Feeling smart and ready to assist you!"
    ],
    r"\bwhat('?s| is) your name\b": [
        "I'm CrescentBot, your virtual assistant for Crescent University.",
        "Call me CrescentBot — I’m here to help you with CUAB-related info!",
        "I'm CrescentBot, nice to meet you!"
    ],
    r"\bwho (are|r) you\b": [
        "I'm a smart assistant built to help students and visitors at Crescent University.",
        "Your Crescent University guide, at your service!",
        "I’m CrescentBot — ask me anything about CUAB."
    ],
    r"\bthank(s| you)\b": [
        "You're welcome!",
        "Happy to help!",
        "Anytime! 😊"
    ],
    r"\bgood (job|bot)\b": [
        "Thank you! I try my best 🤖",
        "Appreciate it!",
        "That means a lot, thanks!"
    ]
}

def is_small_talk(text):
    for pattern in SMALL_TALK_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False

def respond_to_small_talk(text):
    for pattern, responses in SMALL_TALK_PATTERNS.items():
        if re.search(pattern, text, re.IGNORECASE):
            return random.choice(responses)
    return None
