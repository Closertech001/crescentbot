import re
import random

def is_greeting(text):
    greetings = ["hello", "hi", "hey", "good morning", "good afternoon", "good evening"]
    return any(greet in text.lower() for greet in greetings)

def greeting_responses(text=None):
    responses = [
        "Hello! 👋 What would you like to know about Crescent University?",
        "Hi there! 😊 Ask me anything about courses or departments.",
        "Hey! I'm here to help with Crescent University info.",
        "Good to see you! Let me assist with your questions."
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
        "how are you": "I'm great! 😊 Ready to help.",
        "what's up": "Not much! I'm here to answer your questions.",
        "who are you": "I'm Crescent University's virtual assistant 🤖.",
        "are you real": "I'm a helpful bot, yes! 🤖",
        "thank you": "You're welcome!",
        "i love you": "That's sweet! I'm here to support you. ❤️"
    }
    for pattern, reply in responses.items():
        if pattern in text.lower():
            return reply
    return "I'm here to help! Just ask anything about Crescent University."
