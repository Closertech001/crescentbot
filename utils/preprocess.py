import re
from symspellpy.symspellpy import SymSpell, Verbosity
import pkg_resources

# Load SymSpell
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
dictionary_path = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")
sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)

# Abbreviation and slang map
NORMALIZATION_MAP = {
    "uni": "university",
    "cuab": "crescent university",
    "dept": "department",
    "course code": "course",
    "abt": "about",
    "info": "information",
    "u": "you",
    "ur": "your",
    "wat": "what",
    "wats": "what is",
    "pls": "please",
    "plz": "please",
    "cos": "because",
    "bcos": "because",
    "4": "for",
    "2": "to",
    "1st": "first",
    "2nd": "second",
    "3rd": "third",
    "hw": "how"
}

def normalize_input(text: str) -> str:
    text = text.lower()
    for short, full in NORMALIZATION_MAP.items():
        text = re.sub(rf"\b{short}\b", full, text)
    suggestions = sym_spell.lookup_compound(text, max_edit_distance=2)
    if suggestions:
        return suggestions[0].term
    return text
