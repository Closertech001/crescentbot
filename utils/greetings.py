import random
import re
from textblob import TextBlob

# --- GREETING DETECTION ---

GREETING_PATTERNS = [
    r"\bhi\b", r"\bhello\b", r"\bhey\b",
    r"good (morning|afternoon|evening)",
    r"what('?s| is) up", r"\bhowdy\b", r"\byo\b", r"\bsup\b",
    r"\bgreetings\b", r"how far", r"how you dey"
]

_GREETING_RESPONSES_BY_SENTIMENT = {
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

def greeting_response(user_input: str = "") -> str:
    tone = detect_sentiment(user_input) if user_input else "neutral"
    return random.choice(_GREETING_RESPONSES_BY_SENTIMENT.get(tone, _GREETING_RESPONSES_BY_SENTIMENT["neutral"]))


# --- SMALL TALK DETECTION ---

SMALL_TALK_PATTERNS = {
    r"how are you": [
        "I'm doing great, thanks for asking! 😊 How can I help you today?",
        "Feeling sharp and ready to assist! ✨"
    ],
    r"who (are|created|made) you": [
        "I'm the Crescent University Chatbot 🤖, built to help students like you!",
        "I was created to guide you through Crescent Uni life 📘"
    ],
    r"what can you do": [
        "I can help you with course info, departments, fees, and more 🎓",
        "Ask me about admission, courses, or departments — I’ve got answers! 💡"
    ],
    r"tell me about yourself": [
        "I'm a smart little assistant for Crescent University 🧠💬",
        "I answer questions about courses, fees, staff, and more!"
    ],
    r"are you (smart|intelligent)": [
        "I try my best! 😄 Especially when it comes to university questions.",
        "Not bad for a chatbot, right? 😉"
    ],
    r"(you('?| )re|you are) (funny|cool|smart)": [
        "Aww, thanks! 😊 You’re not so bad yourself.",
        "Appreciate it! Let’s keep the good vibes going 🔥"
    ],
    r"thank you|thanks|nice one": [
        "You’re welcome! 😊",
        "Anytime! Let me know if you need more help."
    ]
}

def is_small_talk(user_input: str) -> bool:
    text = user_input.lower()
    return any(re.search(pattern, text) for pattern in SMALL_TALK_PATTERNS)

def small_talk_response(user_input: str) -> str:
    text = user_input.lower()
    for pattern, responses in SMALL_TALK_PATTERNS.items():
        if re.search(pattern, text):
            return random.choice(responses)
    # Default fallback for small talk
    return "I'm here for all your Crescent University questions! 🎓"


# --- COURSE CODE HELPERS ---

def extract_course_code(text: str) -> str | None:
    match = re.search(r"\b([A-Z]{2,4})\s?(\d{3})\b", text.upper())
    if match:
        return f"{match.group(1)} {match.group(2)}"
    return None

def get_course_by_code(course_code: str, course_data: list) -> str | None:
    course_code = course_code.upper().strip()
    for entry in course_data:
        answer = entry.get("answer", "")
        # Check if course code is at the start or inside answer text
        if course_code in answer.upper():
            parts = [part.strip() for part in answer.split(" | ")]
            for part in parts:
                if part.upper().startswith(course_code):
                    return part
    return None
