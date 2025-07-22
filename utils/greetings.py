# utils/greetings.py

import re
import random

GREETING_PATTERNS = [
    r"\bhi\b",
    r"\bhello\b",
    r"\bhey\b",
    r"\bgood (morning|afternoon|evening)\b",
    r"\bwhat'?s up\b",
    r"\bhowdy\b",
    r"\byo\b",
    r"\bsup\b"
]

GREETING_RESPONSES = [
    "Hi there! How can I help you today?",
    "Hello! What would you like to know about Crescent University?",
    "Hey! Ask me anything about departments, courses, or admission.",
    "Greetings! I’m here to help with your Crescent University questions.",
    "Welcome! What information are you looking for?",
    "Hi! I'm your assistant for all things Crescent University."
]

def is_greeting(text: str) -> bool:
    text = text.lower()
    return any(re.search(pattern, text) for pattern in GREETING_PATTERNS)

def get_greeting_response() -> str:
    return random.choice(GREETING_RESPONSES)
