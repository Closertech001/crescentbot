import random
import re
from textblob import TextBlob

# --- GREETING DETECTION ---

GREETING_PATTERNS = [
    r"\bhi\b",
    r"\bhello\b",
    r"\bhey\b",
    r"\bgood (morning|afternoon|evening)\b",
    r"\bwhat's up\b",
    r"\bhowdy\b",
    r"\byo\b",
    r"\bsup\b",
    r"\bgreetings\b",
    r"\bhow far\b",
    r"\bhow you dey\b",
    r"\bwetin dey\b",
    r"\bshey you dey alright\b",
    r"\bhow body\b"
]

_greeting_responses_by_sentiment = {
    "positive": [
        "Hey there! 😊 You sound pumped today. How can I help you shine?",
        "Wetin dey happen? 🌞 You sound happy. Ask me anything jare!",
        "Wassup! 😄 Let’s dive into Crescent Uni gist together!"
    ],
    "neutral": [
        "Hi there! 👋 How can I help you today?",
        "How you dey? 😎 I dey for you anytime, just drop your question.",
        "Yo! 📚 Need anything about Crescent University? Hit me.",
        "Oya na, I'm ready for any school talk you wan run 😁",
        "Hey! Crescent Uni helper at your service 🤓"
    ],
    "negative": [
        "No worry, I got you. 🛠️ Wetin be the issue?",
        "You dey alright? 😔 Talk to me, make we sort am together.",
        "Life fit hard sometimes, but I go help you find answer sharp sharp 💪"
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


# --- SMALL TALK DETECTION ---

SMALL_TALK_PATTERNS = {
    r"how are you": [
        "I dey alright o! 😎 Wetin I go run for you today?",
        "I dey kampe 💯, ready to help with your uni matter!"
    ],
    r"who (are|created|made) you": [
        "Na smart people build me to help Crescent students like you 🤖💡",
        "I be Crescent Uni chatbot — your padi for course and admission wahala 🧠"
    ],
    r"what can you do": [
        "I sabi course info, departments, school fees, admission, and more 🎓",
        "Ask me anything about CUAB, I dey your side 💬"
    ],
    r"tell me about yourself": [
        "I be digital helper for Crescent Uni — sharp like blade! 🔥",
        "My work na to answer school matter, no dull yourself 😎"
    ],
    r"are you (smart|intelligent)": [
        "I try small sha 😁 For Crescent Uni matter, I dey hot!",
        "Omo, na AI I be o! Smart dey my DNA 🤓"
    ],
    r"you('?| )re (funny|cool|smart)": [
        "You sef no dull o! Thanks 🙌",
        "Na you be the real MVP 🏆 I appreciate the vibes!"
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
    return "I dey here to answer all your Crescent Uni questions! 🎓 Just yarn me."


# --- COURSE CODE HELPERS ---

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


# --- FAREWELL DETECTION ---

def detect_farewell(text: str) -> bool:
    farewell_patterns = [
        r"\bbye\b",
        r"\bgoodbye\b",
        r"\bsee you\b",
        r"\blater\b",
        r"\bciao\b",
        r"\btake care\b",
        r"\bcatch you later\b",
        r"\bi dey go\b",
        r"\bmake I dey go\b",
        r"\blater na\b"
    ]
    return any(re.search(pat, text.lower()) for pat in farewell_patterns)


# --- COURSE LEVEL EXPLANATION ---

LEVEL_MEANINGS = {
    "100": "100 level na Year 1 — na your fresher year be that 🎒",
    "200": "200 level na Year 2 — you don dey gather experience 😎",
    "300": "300 level na Year 3 — you don dey deep inside your course 🔍",
    "400": "400 level na Year 4 — final year or almost there 🎓",
    "500": "500 level na for some courses like Law/Engineering — serious final year 🔥",
    "600": "600 level na for advanced courses like Medicine 🩺"
}

FINAL_YEAR_PHRASES = [
    r"final year",
    r"last year",
    r"graduating year",
    r"i.*am.*(in|doing).*final",
    r"i.*dey.*(do|for).*final",
    r"\bi dey graduate\b",
    r"\bi go soon finish\b",
    r"\bi.*(graduate|finish).*soon"
]

def get_level_meaning(text: str) -> str:
    text = text.lower()
    # Check for specific numeric level
    match = re.search(r"\b(100|200|300|400|500|600)\s*level\b", text)
    if match:
        level = match.group(1)
        return LEVEL_MEANINGS.get(level, "")

    # Detect indirect final year references
    for pattern in FINAL_YEAR_PHRASES:
        if re.search(pattern, text):
            # Could return 400 or 500 depending on course — using generic phrasing
            return "Final year fit mean 400 or 500 level — na the last lap be that! 🎓"

    return ""
