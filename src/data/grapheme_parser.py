"""
Bengali grapheme cluster decomposition and analysis.
Deconstructs complex ligatures and words into Root, Vowel Diacritic (Kar), and Consonant Diacritic (Fola).
"""

import unicodedata
from typing import Dict, List, Tuple

# Bengali Unicode ranges and key characters
HASANT = "\u09cd" # ্‌ Virama / Hasant
NUKTA = "\u09bc"  # ় Nukta
REPH = "\u09b0\u09cd" # র্ Reph

VOWEL_KARS = {
    "\u09be": "aa", "\u09bf": "i", "\u09c0": "ee", "\u09c1": "u", "\u09c2": "oo",
    "\u09c3": "rri", "\u09c7": "e", "\u09c8": "oi", "\u09cb": "o", "\u09cc": "ou"
}

FOLAS = {
    "\u09cd\u09af": "ya_fola", # ্য
    "\u09cd\u09b0": "ra_fola", # ্র
    "\u09cd\u09ac": "ba_fola", # ্ব
    "\u09cd\u09ae": "ma_fola", # ্ম
    "\u09cd\u09a8": "na_fola", # ্ন
    "\u09cd\u09b2": "la_fola"  # ্ল
}

def split_grapheme_clusters(word: str) -> List[str]:
    """
    Decomposes a Bengali word into canonical grapheme clusters (akshars).
    """
    norm = unicodedata.normalize("NFC", word.strip())
    clusters = []
    current = []

    i = 0
    while i < len(norm):
        char = norm[i]
        current.append(char)

        # Check if next char is hasant (meaning consonant cluster continues)
        if char == HASANT:
            i += 1
            continue

        # Lookahead: if next character is hasant, don't break cluster
        if i + 1 < len(norm) and norm[i + 1] == HASANT:
            i += 1
            continue

        # If next char is a vowel diacritic or sign (chandrabindu, anusvara, visarga), include it
        if i + 1 < len(norm) and (norm[i + 1] in VOWEL_KARS or norm[i + 1] in ["\u0981", "\u0982", "\u0983"]):
            current.append(norm[i + 1])
            i += 1

        clusters.append("".join(current))
        current = []
        i += 1

    if current:
        clusters.append("".join(current))
    return clusters

def analyze_cluster(cluster: str) -> Dict[str, str]:
    """
    Analyzes a single grapheme cluster and returns its structural components.
    """
    vowel = ""
    fola = ""
    root = cluster

    for kar in VOWEL_KARS:
        if kar in cluster:
            vowel = kar
            root = root.replace(kar, "")

    for f_seq, f_name in FOLAS.items():
        if f_seq in cluster:
            fola = f_name
            root = root.replace(f_seq, "")

    return {
        "cluster": cluster,
        "root": root,
        "vowel": vowel,
        "fola": fola,
        "is_compound": HASANT in root
    }
