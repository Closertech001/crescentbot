import re
import streamlit as st
from symspellpy import SymSpell, Verbosity
import pkg_resources

# Abbreviations for shorthand normalization
ABBREVIATIONS = {
    "u": "you", "r": "are", "ur": "your", "cn": "can", "cud": "could",
    "shud": "should", "wud": "would", "abt": "about", "bcz": "because",
    "plz": "please", "pls": "please", "tmrw": "tomorrow", "wat": "what",
    "wats": "what is", "info": "information", "yr": "year", "sem": "semester",
    "admsn": "admission", "clg": "college", "sch": "school", "uni": "university",
    "cresnt": "crescent", "l": "level", "d": "the", "msg": "message",
    "idk": "i don't know", "imo": "in my opinion", "asap": "as soon as possible",
    "dept": "department", "reg": "registration", "fee": "fees", "pg": "postgraduate",
    "app": "application", "req": "requirement", "nd": "import re
from rapidfuzz import process, fuzz

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
    "nxt": "next", "prev": "previous", "exp": "experience", "std": "student",
    "cos": "because", "cus": "because", "dat": "that", "dis": "this", "fess": "fees",
    "smtn": "something", "smtin": "something", "tins": "things", "abtment": "about me"
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
    "needed for": "required for", "who handles": "who manages", "head of school": "dean",
    "studying": "offering", "taking": "offering", "field": "course", "track": "programme"
}

# --- Pidgin to Standard English ---
PIDGIN_MAP = {
    "wetin": "what", "dey": "is", "una": "you all", "na": "is",
    "abi": "right", "wey": "that", "fit": "can", "go do": "will do",
    "wan": "want", "no get": "don't have", "dey go": "are going",
    "comot": "leave", "carry go": "take away", "waka": "walk", "dey mad": "are you crazy",
    "shey": "do", "how far": "how are you", "no worry": "don’t worry"
}

# --- Apply normalization ---
def normalize_input(text: str) -> str:
    text = text.lower()

    # Phrase replacements (before token split)
    for phrase, replacement in PHRASE_REPLACEMENTS.items():
        text = text.replace(phrase, replacement)

    # Pidgin normalization
    for pidgin, standard in PIDGIN_MAP.items():
        text = text.replace(pidgin, standard)

    # Token-by-token abbreviation replacement
    tokens = text.split()
    normalized_tokens = [ABBREVIATIONS.get(token, token) for token in tokens]
    text = " ".join(normalized_tokens)

    # Synonym replacement
    for word, replacement in SYNONYMS.items():
        text = re.sub(rf"\b{re.escape(word)}\b", replacement, text)

    return text

# --- Optional: fuzzy correction helper ---
def fuzzy_replace(text: str, known_phrases: list, threshold: int = 85) -> str:
    for word in text.split():
        match, score, _ = process.extractOne(word, known_phrases, scorer=fuzz.token_sort_ratio)
        if score >= threshold:
            text = text.replace(word, match)
    return text
national diploma",
    "a-level": "advanced level", "alevel": "advanced level", "2nd": "second",
    "1st": "first", "nxt": "next", "prev": "previous", "exp": "experience",
    "CSC": "department of Computer Science", "Mass comm": "department of Mass Communication",
    "law": "department of law", "Acc": "department of Accounting"
}

# Synonyms to help semantic matching
SYNONYMS = {
    "lecturers": "academic staff", "professors": "academic staff",
    "teachers": "academic staff", "instructors": "academic staff",
    "tutors": "academic staff", "staff members": "staff",
    "head": "dean", "hod": "head of department", "dept": "department",
    "school": "university", "college": "faculty", "course": "subject",
    "class": "course", "subject": "course", "unit": "credit",
    "credit unit": "unit", "course load": "unit", "non teaching": "non-academic",
    "admin worker": "non-academic staff", "support staff": "non-academic staff",
    "clerk": "non-academic staff", "receptionist": "non-academic staff",
    "secretary": "non-academic staff", "tech staff": "technical staff",
    "hostel": "accommodation", "lodging": "accommodation", "room": "accommodation",
    "school fees": "tuition", "acceptance fee": "admission fee", "fees": "tuition",
    "enrol": "apply", "join": "apply", "sign up": "apply", "admit": "apply",
    "requirement": "criteria", "conditions": "criteria", "needed": "required",
    "needed for": "required for", "who handles": "who manages"
}

def get_sym_spell():
    sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
    dictionary_path = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")
    sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
    return sym_spell

def apply_abbreviations(words):
    return [ABBREVIATIONS.get(word, word) for word in words]

def apply_synonyms(words):
    return [SYNONYMS.get(word, word) for word in words]

def normalize_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text

def normalize_input(text):
    text = normalize_text(text)
    words = text.split()
    words = apply_abbreviations(words)
    sym_spell = get_sym_spell()
    corrected = [
        sym_spell.lookup(w, Verbosity.CLOSEST, 2)[0].term if sym_spell.lookup(w, Verbosity.CLOSEST, 2) else w
        for w in words
    ]
    final = apply_synonyms(corrected)
    return " ".join(final)
