import re

# --- Phrase replacements ---
PHRASE_REPLACEMENTS = {
    "biz admin": "business administration",
    "mass comm": "mass communication",
    "cresnt uni": "crescent university",
    "cresnt": "crescent",
    "final year": "final level",
    "last year": "final level",
    "yr 1": "100 level",
    "yr 2": "200 level",
    "yr 3": "300 level",
    "yr 4": "400 level",
    "yr 5": "500 level",
    "year 1": "100 level",
    "year 2": "200 level",
    "year 3": "300 level",
    "year 4": "400 level",
    "year 5": "500 level",
    "1st year": "100 level",
    "2nd year": "200 level",
    "3rd year": "300 level",
    "4th year": "400 level",
    "5th year": "500 level"
}

# --- Abbreviations ---
ABBREVIATIONS = {
    "u": "you", "r": "are", "ur": "your", "cn": "can", "cud": "could",
    "shud": "should", "wud": "would", "abt": "about", "bcz": "because",
    "plz": "please", "pls": "please", "tmrw": "tomorrow", "wat": "what",
    "wats": "what is", "info": "information", "yr": "year", "sem": "semester",
    "admsn": "admission", "clg": "college", "sch": "school", "uni": "university",
    "l": "level", "d": "the", "msg": "message", "dept": "department",
    "reg": "registration", "fee": "fees", "pg": "postgraduate", "app": "application",
    "req": "requirement", "nd": "national diploma", "2nd": "second", "1st": "first",
    "nxt": "next", "prev": "previous", "exp": "experience"
}

# --- Synonyms ---
SYNONYMS = {
    "lecturers": "academic staff", "professors": "academic staff", "teachers": "academic staff",
    "instructors": "academic staff", "tutors": "academic staff", "staff members": "staff",
    "head": "dean", "hod": "head of department", "dept": "department", "school": "university",
    "college": "faculty", "course": "subject", "class": "course", "subject": "course",
    "unit": "credit", "credit unit": "unit", "course load": "unit", "non teaching": "non-academic",
    "admin worker": "non-academic staff", "support staff": "non-academic staff",
    "clerk": "non-academic staff", "receptionist": "non-academic staff",
    "secretary": "non-academic staff", "tech staff": "technical staff",
    "hostel": "accommodation", "lodging": "accommodation", "room": "accommodation",
    "school fees": "tuition", "acceptance fee": "admission fee", "fees": "tuition",
    "enrol": "apply", "join": "apply", "sign up": "apply", "admit": "apply",
    "requirement": "criteria", "conditions": "criteria", "needed": "required",
    "needed for": "required for", "who handles": "who manages"
}

# --- Pidgin normalization ---
PIDGIN_MAP = {
    "wetin": "what", "dey": "is", "una": "you all", "na": "is",
    "abi": "right", "wey": "that", "fit": "can", "go do": "will do",
    "wan": "want", "no get": "don't have", "dey go": "are going",
    "comot": "leave", "carry go": "take away", "waka": "walk"
}

# --- Repeated character reducer (e.g., "heyyyyy" -> "hey") ---
def reduce_repeated_chars(text: str) -> str:
    return re.sub(r'(.)\1{2,}', r'\1', text)

# --- Main normalization function ---
def normalize_input(text: str) -> str:
    # Step 1: Basic cleanup
    text = reduce_repeated_chars(text.lower().strip())

    # Step 2: Phrase-level replacements
    for phrase, replacement in PHRASE_REPLACEMENTS.items():
        text = re.sub(rf"\b{re.escape(phrase)}\b", replacement, text)

    # Step 3: Tokenize and replace
    tokens = re.findall(r"\b\w+\b", text)
    normalized = []

    for token in tokens:
        if token in ABBREVIATIONS:
            normalized.append(ABBREVIATIONS[token])
        elif token in SYNONYMS:
            normalized.append(SYNONYMS[token])
        elif token in PIDGIN_MAP:
            normalized.append(PIDGIN_MAP[token])
        else:
            normalized.append(token)

    return " ".join(normalized)
