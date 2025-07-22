# utils/preprocess.py

import re
import emoji
from symspellpy import SymSpell, Verbosity
import pkg_resources

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

# --- Synonyms for consistent matching ---
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

# --- SymSpell Setup ---
SYM_SPELL = None
def get_sym_spell():
    global SYM_SPELL
    if SYM_SPELL is None:
        SYM_SPELL = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
        dictionary_path = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")
        SYM_SPELL.load_dictionary(dictionary_path, term_index=0, count_index=1)
    return SYM_SPELL

# --- Core cleaning pipeline ---
def normalize_text(text):
    text = text.lower()
    text = emoji.replace_emoji(text, replace='')
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()

def replace_phrases(text):
    for phrase, repl in PHRASE_REPLACEMENTS.items():
        text = re.sub(rf"\b{re.escape(phrase)}\b", repl, text, flags=re.IGNORECASE)
    return text

def apply_abbreviations(words):
    return [ABBREVIATIONS.get(w, w) for w in words]

def apply_pidgin(words):
    return [PIDGIN_MAP.get(w, w) for w in words]

def apply_synonyms(words):
    return [SYNONYMS.get(w, w) for w in words]

def spell_correct(words):
    sym_spell = get_sym_spell()
    corrected = []
    for word in words:
        suggestions = sym_spell.lookup(word, Verbosity.CLOSEST, max_edit_distance=2)
        corrected_word = suggestions[0].term if suggestions else word
        corrected.append(corrected_word)
    return corrected

def convert_final_level(words):
    new_words = []
    for w in words:
        if w == "final":
            new_words.append("400")
            new_words.append("500")
        else:
            new_words.append(w)
    return new_words

# --- Final Pipeline ---
def normalize_input(text):
    text = normalize_text(text)
    text = replace_phrases(text)
    words = text.split()
    words = apply_abbreviations(words)
    words = apply_pidgin(words)
    words = spell_correct(words)
    words = apply_synonyms(words)
    words = convert_final_level(words)
    return " ".join(words)
