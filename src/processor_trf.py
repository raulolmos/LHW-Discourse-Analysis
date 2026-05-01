"""
==============================================================================
MODULE: processor_trf.py
PROJECT: PhD Thesis - Luxury Hospitality Speech Code (LHSC)
VERSION: 3.0 (Substantial Purification)
AUTHOR: [Su Nombre]
SUPERVISOR: Director de Tesis (UCM)

DESCRIPTION:
Advanced NLP pipeline for LHW corpus purification. Implements Transformer-based 
NER masking, Part-of-Speech filtering (focusing on semantic substance), 
and deep domain-specific noise reduction.
==============================================================================
"""

import os
import re
import time
import pandas as pd
import spacy
import torch
import nltk
from nltk.corpus import stopwords
from typing import List

# --- INITIALIZATION & RESOURCE CHECK ---
# GPU acceleration is non-negotiable for Transformer models (Paper 1 quality)
spacy.prefer_gpu()
try:
    nlp = spacy.load("en_core_web_trf")
    print(f"🚀 EXCELLENCE: Transformer model loaded on: {torch.cuda.get_device_name(0)}")
except Exception as e:
    print("❌ ERROR: Failed to load en_core_web_trf. Verify installation.")
    raise

# Ensure NLTK resources are available for standard academic filtering
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

ENG_STOPWORDS = set(stopwords.words('english'))

# --- DOMAIN NOISE & TECHNICAL BLACKLIST ---
# These terms represent 'mediation noise' (UI, legal, technical artifacts) 
# and must be excluded to reveal the 'Luxury Core' of LHW.
DOMAIN_NOISE = {
    # UI/UX & Web Infrastructure
    'website', 'page', 'content', 'site', 'link', 'provider', 'click', 'browse', 
    'browser', 'online', 'internet', 'platform', 'app', 'download', 'user', 
    'visitor', 'navigation', 'menu', 'button', 'scroll', 'availability',
    
    # Transactional & Legal (Residue from web scraping)
    'booking', 'reservation', 'preference', 'advertisement', 'cooky', 'cookie', 
    'datum', 'data', 'policy', 'privacy', 'term', 'condition', 'consent', 
    'manage', 'storage', 'duration', 'necessary', 'third', 'party', 'access',
    
    # Generic hospitality terms (optional: remove if they dilute the LHSC dimensions)
    'stay', 'book', 'room', 'rooms', 'hotel', 'hotel_id', 'available', 'include', 
    'offer', 'offers', 'provide', 'information', 'check', 'type', 'number'
}

# Administrative and structural Regex patterns
ADMIN_PATTERNS = [
    r"(?i)general manager:?.*?\n", # Removes manager blocks
    r"(?i)fax:?.*?(\n|$)",         # Removes fax lines
    r"(?i)code (cin|cir|ciu).*? ", # Removes legal identifiers
    r"https?://\S+|www\.\S+",      # Removes URLs
    r"\S+@\S+",                    # Removes Emails
    r"\d{4,}"                      # Removes long ID numbers/phones
]

def clean_for_substance(text: str) -> str:
    """
    Distills text into pure semantic substance (Nouns & Adjectives).
    Applies NER masking and triple-layer lexical filtering.
    """
    if not isinstance(text, str) or len(text) < 30:
        return ""

    # Phase 1: Structural purification (Regex)
    t = text
    for pattern in ADMIN_PATTERNS:
        t = re.sub(pattern, " ", t)
    
    # Phase 2: NLP Deep Processing
    doc = nlp(t)
    clean_tokens = []
    
    for token in doc:
        # NER Masking: Preserves syntactic structure while neutralizing bias
        if token.ent_type_ in ["GPE", "LOC", "ORG"]:
            # For Sketch Engine context, we could keep tags, 
            # but for pure substance audit, we skip.
            continue 
            
        low_lemma = token.lemma_.lower()
        
        # Phase 3: Triple-Layer Excellence Filter
        # Layer 1: Standard Stopwords & Custom Blacklist
        # Layer 2: Minimum length (avoids noise/fragments)
        # Layer 3: POS Filtering (Only Nouns and Adjectives carry De Barnier's dimensions)
        if (low_lemma not in ENG_STOPWORDS and 
            low_lemma not in DOMAIN_NOISE and 
            len(low_lemma) > 3 and 
            token.pos_ in ['NOUN', 'ADJ']):
            
            clean_tokens.append(low_lemma)
            
    return " ".join(clean_tokens)

def build_lhsc_corpus(csv_path: str, output_dir: str):
    """
    Builds the final corpus with rich metadata in filenames for Paper 2 comparison.
    """
    print(f"📄 Starting Corpus Distillation...")
    df = pd.read_csv(csv_path)
    os.makedirs(output_dir, exist_ok=True)
    
    start_time = time.time()
    count = 0
    
    for _, row in df.iterrows():
        purified = clean_for_substance(str(row['full_narrative']))
        
        # Minimum threshold to ensure scientific validity
        if len(purified.split()) > 10:
            # Metadata naming convention: Ownership_Region_Country_ID
            own = "Indep" if "independent" in str(row['ownership_type']).lower() else "Chain"
            reg = str(row['region']).split()[0].capitalize()
            cou = "".join(str(row['country']).title().split())
            hid = str(row['hotel_id']).upper()
            
            filename = f"{own}_{reg}_{cou}_{hid}.txt"
            with open(os.path.join(output_dir, filename), 'w', encoding='utf-8') as f:
                f.write(purified)
            count += 1
                
    print(f"✅ Processed {count} hotels in {time.time() - start_time:.2f}s.")

def run_substance_audit(output_dir: str, top_n: int = 100):
    """Generates a high-purity frequency audit of the LHSC Core."""
    from collections import Counter
    all_words = []
    for f in os.listdir(output_dir):
        if f.endswith('.txt'):
            with open(os.path.join(output_dir, f), 'r', encoding='utf-8') as file:
                all_words.extend(file.read().split())
    
    print(f"\n💎 THE LHSC LEXICAL CORE (TOP {top_n}):")
    print("-" * 50)
    for i, (word, freq) in enumerate(Counter(all_words).most_common(top_n), 1):
        print(f"{i:3}. {word:15} ({freq:5})")
