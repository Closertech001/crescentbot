# utils/greetings.py

import random
import re
from textblob import TextBlob

# --- GREETING DETECTION ---

GREETING\_PATTERNS = \[
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

\_greeting\_responses\_by\_sentiment = {
"positive": \[
"Hey there! 😊 You're sounding great today. How can I assist you?",
"Hi! 👋 I'm glad you're feeling good. What would you like to know?",
"Hello! 🌟 Ready to explore Crescent University together?"
],
"neutral": \[
"Hi there! 😊 How can I help you?",
"Hello! 👋 What would you like to know about Crescent University?",
"Hey! I'm here to assist you with your course or university questions.",
"Hi! Let me know what you're looking for.",
"How far! I'm here for any Crescent Uni gist you need."
],
"negative": \[
"I'm here to help — let’s figure it out together. 💡",
"Sorry if you're having a rough time. Let's fix that. What do you need?",
"I’ve got your back. Let me help you with that. 💪"
]
}

def is\_greeting(user\_input: str) -> bool:
text = user\_input.lower()
return any(re.search(pattern, text) for pattern in GREETING\_PATTERNS)

def detect\_sentiment(user\_input: str) -> str:
analysis = TextBlob(user\_input)
if analysis.sentiment.polarity > 0.2:
return "positive"
elif analysis.sentiment.polarity < -0.2:
return "negative"
return "neutral"

def greeting\_responses(user\_input: str = "") -> str:
tone = detect\_sentiment(user\_input) if user\_input else "neutral"
return random.choice(\_greeting\_responses\_by\_sentiment.get(tone, \_greeting\_responses\_by\_sentiment\["neutral"]))

# --- SMALL TALK DETECTION ---

SMALL\_TALK\_PATTERNS = {
r"how are you": \[
"I'm doing great, thanks for asking! 😊 How can I help you today?",
"Feeling sharp and ready to assist! ✨"
],
r"who (are|created|made) you": \[
"I'm the Crescent University Chatbot 🤖, built to help students like you!",
"I was created to guide you through Crescent Uni life 📘"
],
r"what can you do": \[
"I can help you with course info, departments, fees, and more 🎓",
"Ask me about admission, courses, or departments — I’ve got answers! 💡"
],
r"tell me about yourself": \[
"I'm a smart little assistant for Crescent University 🧠💬",
"I answer questions about courses, fees, staff, and more!"
],
r"are you (smart|intelligent)": \[
"I try my best! 😄 Especially when it comes to university questions.",
"Not bad for a chatbot, right? 😉"
],
r"you('?| )re (funny|cool|smart)": \[
"Aww, thanks! 😊 You’re not so bad yourself.",
"Appreciate it! Let’s keep the good vibes going 🔥"
]
}

def is\_small\_talk(user\_input: str) -> bool:
text = user\_input.lower()
return any(re.search(pattern, text) for pattern in SMALL\_TALK\_PATTERNS)

def small\_talk\_response(user\_input: str) -> str:
text = user\_input.lower()
for pattern, responses in SMALL\_TALK\_PATTERNS.items():
if re.search(pattern, text):
return random.choice(responses)
return "I'm here for all your Crescent University questions! 🎓"

# --- COURSE CODE HELPERS ---

def extract\_course\_code(text: str) -> str:
match = re.search(r"\b(\[A-Z]{2,4})\s?(\d{3})\b", text.upper())
if match:
return f"{match.group(1)} {match.group(2)}"
return None

def get\_course\_by\_code(course\_code: str, course\_data: list) -> str:
course\_code = course\_code.upper().strip()
for entry in course\_data:
if course\_code in entry.get("answer", ""):
parts = \[part.strip() for part in entry\["answer"].split(" | ")]
for part in parts:
if part.startswith(course\_code):
return part
return None
