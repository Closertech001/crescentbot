import random
import re

# Simple patterns for detecting greetings
GREETING_KEYWORDS = [
    "hello", "hi", "hey", "good morning", "good afternoon", "good evening", "greetings", "howdy"
]

GREETING_RESPONSES = [
    "Hello! 👋 How can I assist you with Crescent University today?",
    "Hi there! 😊 What would you like to know about Crescent University?",
    "Hey! 👋 Feel free to ask me anything related to Crescent University.",
    "Greetings! 🤝 I’m here to help with your Crescent University questions."
]

def is_greeting(text):
    text = text.lower()
    return any(re.search(rf"\b{kw}\b", text) for kw in GREETING_KEYWORDS)

def greeting_responses(text=None):
    return random.choice(GREETING_RESPONSES)
