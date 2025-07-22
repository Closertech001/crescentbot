import re
from symspellpy.symspellpy import SymSpell, Verbosity
import pkg_resources
from rapidfuzz import fuzz, process

# --- INIT SPELL CHECKER ---
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
dictionary_path = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")
sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)

# --- ABBREVIATION AND SYNONYM MAP ---
NORMALIZATION_MAP = {
    "cs": "computer science",
    "mass comm": "mass communication",
    "comm": "communication",
    "law dept": "law",
    "archi": "architecture",
    "med": "medicine",
    "microbio": "microbiology",
    "physio": "physiology",
    "nursing sci": "nursing",
    "ict": "information and communication technology",
    "ict dept": "information and communication technology",
    "env sci": "environmental science",
    "cohes": "college of health sciences",
    "coict": "college of information and communication technology",
    "coes": "college of environmental sciences",
    "conas": "college of natural and applied sciences",
    "casmas": "college of arts, social and management sciences",
    "bacolaw": "bola ajibola college of law",
    "cuab": "crescent university"
}

SYNONYMS = {
    "requirements": "requirement",
    "admission criteria": "admission requirement",
    "study": "course",
    "courses": "course",
    "subject": "course",
    "unit": "units",
    "credit": "units",
    "lecturer": "teacher",
    "hod": "head of department"
}

# --- FUZZY MATCH CANDIDATES ---
FUZZY_CANDIDATES = list(NORMALIZATION_MAP.keys()) + list(SYNONYMS.keys())

def expand_abbreviation(text: str) -> str:
    for short, full in NORMALIZATION_MAP.items():
        pattern = r'\b' + re.escape(short) + r'\b'
        text = re.sub(pattern, full, text, flags=re.IGNORECASE)
    return text

def replace_synonyms(text: str) -> str:
    for syn, standard in SYNONYMS.items():
        pattern = r'\b' + re.escape(syn) + r'\b'
        text = re.sub(pattern, standard, text, flags=re.IGNORECASE)
    return text

def correct_spelling(text: str) -> str:
    words = text.split()
    corrected_words = []
    for word in words:
        suggestions = sym_spell.lookup(word, Verbosity.CLOSEST, max_edit_distance=2)
        if suggestions:
            corrected_words.append(suggestions[0].term)
        else:
            corrected_words.append(word)
    return ' '.join(corrected_words)

def fuzzy_normalize(text: str) -> str:
    words = text.split()
    updated = []
    for word in words:
        match, score, _ = process.extractOne(word, FUZZY_CANDIDATES, scorer=fuzz.ratio)
        if score >= 85:
            # Replace with mapped normalized value if available
            if match in NORMALIZATION_MAP:
                updated.append(NORMALIZATION_MAP[match])
            elif match in SYNONYMS:
                updated.append(SYNONYMS[match])
            else:
                updated.append(match)
        else:
            updated.append(word)
    return ' '.join(updated)

def normalize_input(user_input: str) -> str:
    """Main normalization pipeline"""
    text = user_input.lower()
    text = expand_abbreviation(text)
    text = replace_synonyms(text)
    text = correct_spelling(text)
    text = fuzzy_normalize(text)
    return text
