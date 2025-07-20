# utils/greetings.py

import random
import re
from textblob import TextBlob

GREETING_PATTERNS = [
    r"hi",
    r"hello",
    r"hey",
    r"good (morning|afternoon|evening)",
    r"what's up",
    r"howdy",
    r"yo",
    r"sup",
    r"greetings",
    r"how far",
    r"how you dey"
]

greeting_responses = [
    "Hello there! 😊 How can I assist you today?",
    "Hi! 👋 What would you like to know about Crescent University?",
    "Hey! I’m here to help. Ask me anything!",
    "Greetings! Ready to explore Crescent University?",
    "How far! I'm here for any Crescent Uni gist you need."
]

def is_greeting(user_input: str) -> bool:
    """Returns True if user input is a greeting."""
    text = user_input.lower()
    return any(re.search(pattern, text) for pattern in GREETING_PATTERNS)

def detect_sentiment(user_input: str) -> str:
    """Classifies sentiment of the input as 'positive', 'neutral', or 'negative'."""
    analysis = TextBlob(user_input)
    if analysis.sentiment.polarity > 0.2:
        return "positive"
    elif analysis.sentiment.polarity < -0.2:
        return "negative"
    return "neutral"

def greeting_responses():
    return random.choice(greeting_responses)
