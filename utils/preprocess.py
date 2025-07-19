import re
from symspellpy import SymSpell, Verbosity
import pkg_resources
import streamlit as st

# Common abbreviations
ABBREVIATIONS = {
    "u": "you", "r": "are", "ur": "your", "pls": "please", "plz": "please",
    "tmrw": "tomorrow", "cn": "can", "wat": "what", "abt": "about", "bcz": "because",
    "dept": "department", "admsn": "admission", "uni": "university", "sch": "school",
    "info": "information", "l": "level", "sem": "semester"
}

# Optional mapping (for future use)
DEPARTMENT_MAP = {
    "CSC": "Computer Science", "ECO": "Economics with Operations Research", "LAW": "Law (BACOLAW)",
    "ANA": "Anatomy", "PIS": "Political Science", "GST": "General Studies", "PHY": "Physics"
}

# Load and cache SymSpell
@st.cache_resource
def get_sym_spell():
    sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
    dictionary_path = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")
    sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
    return sym_spell

# Normalize text: lowercase, remove extra chars
def normalize_text(text):
    text = re.sub(r'[^\w\s\-]', '', text)          # keep alphanum + hyphen
    text = re.sub(r'(.)\1{2,}', r'\1', text)       # limit repeated letters
    return text.lower()

# Preprocess text: normalize, expand, correct
def preprocess_text(text, debug=False):
    text = normalize_text(text)
    words = text.split()
    expanded = [ABBREVIATIONS.get(w.lower(), w) for w in words]
    
    sym_spell = get_sym_spell()
    corrected = []
    for word in expanded:
        suggestions = sym_spell.lookup(word, Verbosity.CLOSEST, max_edit_distance=2)
        corrected.append(suggestions[0].term if suggestions else word)

    if debug:
        print("Raw:", text)
        print("Expanded:", expanded)
        print("Corrected:", corrected)

    return ' '.join(corrected)

# Extract course prefix (e.g., CSC from CSC-101)
def extract_prefix(code):
    match = re.match(r"([A-Z\-]+)", code)
    return match.group(1) if match else None
