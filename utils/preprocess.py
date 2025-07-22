import re
from symspellpy import SymSpell, Verbosity
import pkg_resources

ABBREVIATIONS = {
    "comp sci": "computer science",
    "com sci": "computer science",
    "compsci": "computer science",
    "mass comm": "mass communication",
    "masscom": "mass communication",
    "med lab": "medical laboratory science",
    "medlab": "medical laboratory science",
    "nurs": "nursing",
    "nursin": "nursing",
    "biz admin": "business administration",
    "int'l rel": "international relations",
    "int rel": "international relations",
    "irs": "international relations",
    "econs": "economics with operations research",
    "pol sci": "political science",
    "pis": "political science",
    "mls": "medical laboratory science",
    "cs": "computer science",
    "ir": "international relations",
}

SYNONYMS = {
    "dept": "department",
    "uni": "university",
    "school": "university",
    "courses": "course",
    "requirement": "requirements",
    "admission": "admission",
    "fees": "fee",
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
