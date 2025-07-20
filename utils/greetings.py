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

_greeting_responses_by_sentiment = {
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

def greeting_responses(user_input: str = "") -> str:
    tone = detect_sentiment(user_input) if user_input else "neutral"
    return random.choice(_greeting_responses_by_sentiment.get(tone, _greeting_responses_by_sentiment["neutral"]))

# --- NEW: Course code helpers ---

def extract_course_code(text: str) -> str:
    match = re.search(r"\b([A-Z]{2,4})\s?(\d{3})\b", text.upper())
    if match:
        return f"{match.group(1)} {match.group(2)}"
    return None

def get_course_by_code(course_code: str, course_data: list) -> str:
    course_code = course_code.upper().strip()
    for entry in course_data:
        if course_code in entry.get("answer", ""):
            parts = [part.strip() for part in entry["answer"].split(" | ")]
            for part in parts:
                if part.startswith(course_code):
                    return part
    return None
