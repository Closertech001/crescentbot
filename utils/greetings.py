import re
import random

def is_greeting(text):
    greetings = ["hello", "hi", "good morning", "good afternoon", "good evening", "hey"]
    return any(greet in text.lower() for greet in greetings)

def greeting_responses(_):
    responses = [
        "Hello! 👋 What would you like to know about Crescent University?",
        "Hi there! 😊 How can I help you today?",
        "Hey! Feel free to ask anything about CUAB.",
        "Welcome! What do you want to explore about Crescent University?"
    ]
    return random.choice(responses)

def is_small_talk(text):
    patterns = [
        r"\bhow are you\b", r"\bwhat's up\b", r"\bare you a robot\b", r"\bwho made you\b",
        r"\bwhat can you do\b", r"\bwho are you\b", r"\btell me about yourself\b"
    ]
    return any(re.search(p, text.lower()) for p in patterns)

def small_talk_response(text):
    if "how are you" in text.lower():
        return random.choice(["I'm doing great! 😄", "All systems go! How about you?"])
    elif "who made you" in text.lower():
        return "I was developed by a student using OpenAI technology and Crescent University data."
    elif "what can you do" in text.lower():
        return "I can answer questions about departments, courses, fees, and admission info at Crescent University."
    elif "are you a robot" in text.lower():
        return "Not quite! I'm a smart chatbot trained to help with Crescent University inquiries."
    else:
        return "I'm here to help with any Crescent University-related questions!"

def extract_course_code(text):
    match = re.search(r"\b([A-Z]{2,4}[-\s]?\d{3})\b", text)
    return match.group(1).replace(" ", "").upper() if match else None

def get_course_by_code(code, course_data):
    code = code.replace(" ", "").upper()
    for entry in course_data:
        if entry["code"].replace(" ", "").upper() == code:
            return entry["answer"]
    return None
