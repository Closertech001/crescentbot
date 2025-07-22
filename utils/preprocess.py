import re
import emoji
import streamlit as st
from symspellpy import SymSpell, Verbosity
import pkg_resources

# Phrase-based replacements (multi-word first to preserve intent)
PHRASE_REPLACEMENTS = {
    "biz admin": "business administration",
    "mass comm": "mass communication",
    "comp sci": "computer science",
    "cresnt uni": "crescent university",
    "lagos campus": "campus in lagos"
}

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
    "app": "application", "req": "requirement", "nd": "national diploma",
    "alevel": "advanced level", "2nd": "second", "1st": "first", "nxt": "next",
    "prev": "previous", "exp": "experience"
}

# Synonyms to improve semantic matching
SYNONYMS = {
    "lecturers": "academic staff", "professors": "academic staff",
    "teachers": "academic staff", "instructors": "academic staff",
    "tutors": "academic staff", "staff members": "staff",
    "head": "dean", "hod": "head of department", "school": "university",
    "college": "faculty", "course": "subject", "class": "course",
    "unit": "credit", "credit unit": "unit", "course load": "unit",
    "non teaching": "non-academic", "admin worker": "non-academic staff",
    "support staff": "non-academic staff", "clerk": "non-academic staff",
    "receptionist": "non-academic staff", "secretary": "non-academic staff",
    "tech staff": "technical staff", "hostel": "accommodation",
    "lodging": "accommodation", "room": "accommodation", "school fees": "tuition",
    "acceptance fee": "admission fee", "fees": "tuition", "enrol": "apply",
    "join": "apply", "sign up": "apply", "admit": "apply",
    "requirement": "criteria", "conditions": "criteria",
    "needed for": "required for", "who handles": "who manages",
    "who is in charge of": "who manages"
}

# Basic Nigerian Pidgin replacements
PIDGIN_MAP = {
    "wetin": "what", "dey": "is", "una": "you all", "na": "is",
    "abi": "right", "wey": "that", "fit": "can", "go do": "will do",
    "wan": "want", "no get": "don't have", "dey go": "are going",
    "comot": "leave", "carry go": "take away", "waka": "walk"
}

# SymSpell singleton for spell correction
SYM_SPELL = None

def get_sym_spell():
    global SYM_SPELL
    if SYM_SPELL is None:
        SYM_SPELL = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
        dictionary_path = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")
        SYM_SPELL.load_dictionary(dictionary_path, term_index=0, count_index=1)
    return SYM_SPELL

# Text cleaning
def normalize_text(text):
    text = text.lower()
    text = emoji.replace_emoji(text, replace='')  # remove emoji
    text = re.sub(r"[^\w\s]", "", text)  # remove punctuation
    return text.strip()

# Phrase-level replacement (e.g., "biz admin" → "business administration")
def replace_phrases(text):
    for phrase, repl in PHRASE_REPLACEMENTS.items():
        pattern = re.compile(rf"\b{re.escape(phrase)}\b", re.IGNORECASE)
        text = pattern.sub(repl, text)
    return text

def apply_abbreviations(words):
    return [ABBREVIATIONS.get(word, word) for word in words]

def apply_pidgin(words):
    return [PIDGIN_MAP.get(word, word) for word in words]

def apply_synonyms(words):
    return [SYNONYMS.get(word, word) for word in words]

def spell_correct(words):
    sym_spell = get_sym_spell()
    corrected = []
    for word in words:
        suggestions = sym_spell.lookup(word, Verbosity.CLOSEST, max_edit_distance=2)
        corrected_word = suggestions[0].term if suggestions else word
        corrected.append(corrected_word)
    return corrected

# Main normalization pipeline
def normalize_input(text):
    text = normalize_text(text)
    text = replace_phrases(text)
    words = text.split()
    words = apply_abbreviations(words)
    words = apply_pidgin(words)
    words = spell_correct(words)
    words = apply_synonyms(words)
    return " ".join(words)
