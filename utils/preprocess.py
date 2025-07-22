import re
import emoji
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
    "app": "application", "req": "requirement", "nd": "national diploma",
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

PIDGIN_MAP = {
    "wetin": "what",
    "dey": "is",
    "una": "you all",
    "na": "is",
    "abi": "right",
    "wey": "that",
    "fit": "can",
    "go do": "will do",
    "wan": "want",
    "no get": "don't have",
    "dey go": "are going",
    "comot": "leave",
    "carry go": "take away",
    "waka": "walk"
}

# --- SymSpell Singleton ---
SYM_SPELL = None

def get_sym_spell():
    global SYM_SPELL
    if SYM_SPELL is None:
        SYM_SPELL = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
        dictionary_path = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")
        SYM_SPELL.load_dictionary(dictionary_path, term_index=0, count_index=1)
    return SYM_SPELL

# --- Cleaning ---
def normalize_text(text):
    text = text.lower()
    text = emoji.replace_emoji(text, replace='')  # Remove emojis
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()

def apply_abbreviations(words):
    return [ABBREVIATIONS.get(word, word) for word in words]

def apply_pidgin(words):
    return [PIDGIN_MAP.get(word, word) for word in words]

def apply_synonyms(words):
    return [SYNONYMS.get(word, word) for word in words]

def normalize_input(text):
    text = normalize_text(text)
    words = text.split()
    words = apply_abbreviations(words)
    words = apply_pidgin(words)
    sym_spell = get_sym_spell()
    corrected = [
        sym_spell.lookup(w, Verbosity.CLOSEST, 2)[0].term if sym_spell.lookup(w, Verbosity.CLOSEST, 2) else w
        for w in words
    ]
    final = apply_synonyms(corrected)
    return " ".join(final)
