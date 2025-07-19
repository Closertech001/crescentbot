import re
from symspellpy import SymSpell, Verbosity
import pkg_resources

ABBREVIATIONS = {
    "u": "you", "r": "are", "ur": "your", "pls": "please", "plz": "please",
    "tmrw": "tomorrow", "cn": "can", "wat": "what", "abt": "about", "bcz": "because",
    "dept": "department", "admsn": "admission", "uni": "university", "sch": "school",
    "info": "information", "l": "level", "sem": "semester"
}

DEPARTMENT_MAP = {
    "CSC": "Computer Science", "ECO": "Economics with Operations Research", "LAW": "Law (BACOLAW)",
    "ANA": "Anatomy", "PIS": "Political Science", "GST": "General Studies", "PHY": "Physics"
}

sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
dictionary_path = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")
sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)

def normalize_text(text):
    text = re.sub(r'([^a-zA-Z0-9\s])', '', text)
    text = re.sub(r'(.)\1{2,}', r'\1', text)
    return text.lower()

def preprocess_text(text):
    text = normalize_text(text)
    words = text.split()
    expanded = [ABBREVIATIONS.get(w.lower(), w) for w in words]
    corrected = []
    for word in expanded:
        suggestions = sym_spell.lookup(word, Verbosity.CLOSEST, max_edit_distance=2)
        corrected.append(suggestions[0].term if suggestions else word)
    return ' '.join(corrected)

def extract_prefix(code):
    match = re.match(r"([A-Z\-]+)", code)
    return match.group(1) if match else None
