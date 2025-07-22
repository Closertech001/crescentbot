import re
import os
from symspellpy import SymSpell, Verbosity
import pkg_resources

# --- INIT SPELL CHECKER ---
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
dictionary_path = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")
sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)

# --- SLANG/SYNONYM NORMALIZATION ---
NORMALIZATION_MAP = {
    "biz admin": "business administration",
    "mass comm": "mass communication",
    "comp sci": "computer science",
    "comp. sci.": "computer science",
    "med lab": "medical laboratory science",
    "elect elect": "electrical and electronics engineering",
    "econs": "economics",
    "english lang": "english language",
    "final year": "400 level",
    "year one": "100 level",
    "year two": "200 level",
    "year three": "300 level",
    "year four": "400 level",
    "year five": "500 level",
    "400 level": "400",
    "500 level": "500",
    "300 level": "300",
    "200 level": "200",
    "100 level": "100",
}

def normalize_text(text: str) -> str:
    """
    Normalize user input using synonym/slang replacement and spell correction.
    """
    original_text = text.lower()
    
    # Replace known synonyms/slangs
    for pattern, replacement in NORMALIZATION_MAP.items():
        original_text = re.sub(rf"\b{re.escape(pattern)}\b", replacement, original_text)

    # Spell correction
    corrected_words = []
    for word in original_text.split():
        suggestions = sym_spell.lookup(word, Verbosity.CLOSEST, max_edit_distance=2)
        corrected_words.append(suggestions[0].term if suggestions else word)

    return " ".join(corrected_words)
