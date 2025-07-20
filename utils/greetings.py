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

greeting_responses_by_sentiment = {
    "positive": [
        "Hey there! 😊 You're sounding great today. How can I assist you?",
        "Hi! 👋 I'm glad you're feeling good. What would you like to know?",
        "Hello! 🌟 Ready to explore Crescent University together?"
    ],
    "neutral": [
        "Hi there! 😊 How can I help you?",
        "Hello! 👋 What would you like to know about Crescent University?",
        "Hey! I'm here to assist you with your course or university questions.",
        "Hi! Let me know what you're looking for.",
        "How far! I'm here for any Crescent Uni gist you need."
    ],
    "negative": [
        "I'm here to help — let’s figure it out together. 💡",
        "Sorry if you're having a rough time. Let's fix that. What do you need?",
        "I’ve got your back. Let me help you with that. 💪"
    ]
}

def is_greeting(user_input: str) -> bool:
    text = user_input.lower()
    return any(re.search(pattern, text) for pattern in GREETING_PATTERNS)

def detect_sentiment(user_input: str) -> str:
    analysis = TextBlob(user_input)
    if analysis.sentiment.polarity > 0.2:
        return "positive"
    elif analysis.sentiment.polarity < -0.2:
        return "negative"
    return "neutral"

def greeting_responses(user_input: str) -> str:
    tone = detect_sentiment(user_input)
    return random.choice(greeting_responses_by_sentiment.get(tone, greeting_responses_by_sentiment["neutral"]))
