# utils/preprocess.py

import os
import re
import json
from symspellpy.symspellpy import SymSpell, Verbosity

# Setup SymSpell
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
dictionary_path = os.path.join("data", "frequency_dictionary_en_82_765.txt")
if os.path.exists(dictionary_path):
    sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)

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

# --- Abbreviations for shorthand normalization ---
abbreviation_map = {
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

# --- Synonyms for consistent matching ---
synonym_map = {
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

def normalize_text(text: str) -> str:
    # Expand abbreviations
    for abbr, full in abbreviation_map.items():
        pattern = re.compile(rf"\b{re.escape(abbr)}\b", re.IGNORECASE)
        text = pattern.sub(full, text)

    # Replace synonyms
    for word, synonym in synonym_map.items():
        pattern = re.compile(rf"\b{re.escape(word)}\b", re.IGNORECASE)
        text = pattern.sub(synonym, text)

    return text

def spell_correct(text: str) -> str:
    words = text.split()
    corrected_words = []
    for word in words:
        suggestions = sym_spell.lookup(word, Verbosity.CLOSEST, max_edit_distance=2)
        corrected_words.append(suggestions[0].term if suggestions else word)
    return " ".join(corrected_words)

def detect_level_from_text(text: str) -> str:
    if "final year" in text:
        return "400"  # Or "500" depending on department
    match = re.search(r"\b([1-5]00)\b", text)
    return match.group(1) if match else ""

def preprocess_input(user_input: str) -> dict:
    original = user_input.lower().strip()
    normalized = normalize_text(original)
    corrected = spell_correct(normalized)

    query_info = {"original": original, "normalized": corrected}

    level = detect_level_from_text(corrected)
    if level:
        query_info["level"] = level

    return query_info
