# utils/greetings.py

def detect_greeting(user_input):
    greetings = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"]
    input_cleaned = user_input.lower().strip()
    return input_cleaned in greetings


def generate_greeting_response():
    return "Hello! How can I help you with Crescent University today?"


def is_small_talk(user_input):
    small_talk_phrases = [
        "how are you", "what's up", "how's it going", 
        "tell me a joke", "who are you", "are you real", 
        "are you human", "can you help", "just testing"
    ]
    input_cleaned = user_input.lower().strip()

    # Only match small talk if it's short and not a specific query
    if len(input_cleaned.split()) < 7:
        return any(phrase in input_cleaned for phrase in small_talk_phrases)
    
    return False


def generate_small_talk_response():
    return random.choice([
        "I'm just a helpful chatbot, but I'm doing great! 😊",
        "Here to help! What would you like to know about Crescent University?",
        "Ask me anything — I'm ready to assist you!"
    ])
