import re
import random

def is_greeting(text):
    greetings = ["hello", "hi", "hey", "good morning", "good afternoon", "good evening", "greetings", "what's up"]
    return any(greet in text.lower() for greet in greetings)

def greeting_responses(text=None):
    responses = [
        "Hello! 👋 What would you like to know about Crescent University?",
        "Hi there! 😊 Feel free to ask me about courses, departments or anything else.",
        "Hey! I'm here to help with your Crescent University questions.",
        "Good to see you! Ready to answer your university questions."
    ]
    return random.choice(responses)

def is_small_talk(text):
    patterns = [
        r"how (are|r) (you|u)",
        r"what'?s up",
        r"how'?s it going",
        r"who (are|r) you",
        r"are you (a bot|real)",
        r"thank(s| you)",
        r"(i love|like) you",
        r"good (job|work)"
    ]
    return any(re.search(p, text.lower()) for p in patterns)

def small_talk_response(text):
    responses = {
        "how are you": "I'm doing great, thanks! How can I assist you today?",
        "what's up": "Not much, just here to help you!",
        "who are you": "I'm your Crescent University assistant bot!",
        "are you real": "I'm a real bot 🤖 trained to answer your questions.",
        "thank you": "You're welcome! 😊",
        "i love you": "Aww, I love helping you too! ❤️"
    }

    for pattern, reply in responses.items():
        if pattern in text.lower():
            return reply

    return "I'm here to help! Just ask me anything about Crescent University."
