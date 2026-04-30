"""
LHSC (Luxury Hospitality Speech Code) - Corpus Analyzer
This module performs a highly detailed lexical audit and maps hotel 
narratives to the theoretical dimensions proposed by De Barnier et al.
"""

import os
import re
import pandas as pd
import spacy

# Load the NLP model for English
# Rationale: 'en_core_web_sm' provides an optimal balance between accuracy 
# and computational efficiency for POS tagging and lemmatization.
nlp = spacy.load('en_core_web_sm')

# --- THEORETICAL FRAMEWORK MAPPING ---
# Rationale: Seeds defining De Barnier et al.'s dimensions adapted for Hospitality.
DIMENSIONS_MAP = {
    'Heritage': ['historic', 'palazzo', 'ancient', 'legacy', 'heritage', 'established', 'century', 'landmark'],
    'Hedonism': ['experience', 'sensory', 'pleasure', 'unforgettable', 'indulge', 'serene', 'wellness', 'oasis'],
    'Elitism': ['exclusive', 'private', 'discreet', 'member', 'elite', 'prestigious', 'sanctuary'],
    'Savoir_faire': ['bespoke', 'curated', 'handcrafted', 'excellence', 'meticulous', 'artisan', 'tailored']
}

def analyze_hotel_narrative(file_path):
    """
    Reads, cleans, and semantically profiles a single hotel narrative.
    Returns metrics on text purity and dimensional keyword frequencies.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        raw_text = f.read()

    # 1. PRE-CLEANING METRICS (Integrity Audit)
    pre_word_count = len(raw_text.split())

    # 2. EXHAUSTIVE PURGE (UI/UX Artifact Removal)
    # Rationale: Stripping transactional and navigational noise to isolate pure brand discourse.
    ui_noise = [
        r'skip to (primary )?content', r'check availability', r'book now', 
        r'local time', r'weather \d+.*?c', r'navigation', r'view virtual tour'
    ]
    
    clean_text = raw_text
    for pattern in ui_noise:
        clean_text = re.sub(pattern, '', clean_text, flags=re.IGNORECASE)

    # 3. NLP PROCESSING (Tokenization, Lemmatization, Filtering)
    doc = nlp(clean_text)
    
    # Rationale: Keep only alphabetic, non-stopword tokens longer than 2 chars.
    tokens = [token.lemma_.lower() for token in doc 
              if token.is_alpha and not token.is_stop and len(token.text) > 2]
    
    post_word_count = len(tokens)
    
    # 4. DIMENSIONAL MAPPING (De Barnier vs. LHSC)
    scores = {dim: 0 for dim in DIMENSIONS_MAP.keys()}
    
    for token in tokens:
        for dim, keywords in DIMENSIONS_MAP.items():
            if token in keywords:
                scores[dim] += 1

    # Safely calculate retention rate to avoid division by zero
    retention_rate = (post_word_count / pre_word_count) if pre_word_count > 0 else 0

    return {
        'pre_len': pre_word_count,
        'post_len': post_word_count,
        'purity_index': retention_rate,
        **scores
    }

def process_entire_corpus(corpus_path):
    """
    Iterates over the processed corpus directory and compiles the final dataset.
    """
    results = []
    
    for file in os.listdir(corpus_path):
        if file.endswith('.txt'):
            full_path = os.path.join(corpus_path, file)
            analysis = analyze_hotel_narrative(full_path)
            
            # Extract metadata from filename (e.g., ITA_IND_001.txt)
            res = {
                'hotel_id': file,
                'region': file[:3]
            }
            res.update(analysis)
            results.append(res)
            
    return pd.DataFrame(results)

# This allows the script to be imported without running automatically
if __name__ == "__main__":
    print("LHSC Analyzer ready to be imported.")
