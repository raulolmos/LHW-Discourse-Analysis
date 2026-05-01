"""
==============================================================================
Module: processor_trf.py
Project: PhD Thesis - Luxury Hospitality Speech Code (LHSC)
Focus: Paper 1 - Instrument Validation & Corpus Distillation
Institution: Universidad Complutense de Madrid (UCM)

Description:
Advanced NLP pipeline using spaCy Transformers for LHW corpus purification.
Implements NER Masking to preserve syntactic context for Sketch Engine 
while neutralizing geographic and brand-specific frequency bias.
==============================================================================
"""

import os
import re
import time
import pandas as pd
import spacy
import torch
from typing import List

# Ensure GPU usage for Transformer efficiency
spacy.prefer_gpu()
try:
    nlp = spacy.load("en_core_web_trf")
    print(f"✅ Transformer model loaded on: {torch.cuda.get_device_name(0)}")
except Exception as e:
    print("❌ Error loading en_core_web_trf. Ensure it is installed.")
    raise

# UI and Administrative noise patterns (Refined from Sources 1, 2, 3)
ADMIN_PATTERNS = [
    r"(?i)general manager:?.*?\n",
    r"(?i)fax:?.*?(\n|$)",
    r"(?i)code (cin|cir|ciu).*? ",
    r"(?i)official browser support.*?browser",
    r"(?i)this browser is not supported.*",
    r"https?://\S+|www\.\S+",  # URLs
    r"\S+@\S+",                  # Emails
    r"\d{4,}"                    # Long ID numbers/Phones
]

# Navigational/Functional Stopwords to be removed even if they are valid POS
DOMAIN_UI_STOPWORDS = {
    "signup", "login", "discover", "view", "menu", "availability", 
    "online", "account", "join", "click", "more", "scroll"
}

def clean_and_mask_text(text: str) -> str:
    """
    Purifies text by removing admin noise and masking entities 
    to preserve syntax for Sketch Engine analysis.
    """
    if not isinstance(text, str) or len(text) < 20:
        return ""

    # Phase 1: Structural Regex Cleaning
    t = text
    for pattern in ADMIN_PATTERNS:
        t = re.sub(pattern, " ", t)
    
    # Phase 2: NLP Masking & Tokenization
    doc = nlp(t)
    clean_tokens = []
    
    for token in doc:
        # Entity Masking: Replaces names with generic tags to avoid bias[cite: 3, 4]
        if token.ent_type_ in ["GPE", "LOC"]:
            clean_tokens.append("LOCATION_ENTITY")
            continue
        if token.ent_type_ in ["ORG"]:
            clean_tokens.append("BRAND_ENTITY")
            continue
            
        # UI Noise Filtering
        if token.lemma_.lower() in DOMAIN_UI_STOPWORDS:
            continue
            
        # Basic filtering: Keep only meaningful words but preserve structure
        if token.is_bracket or token.like_url or token.is_space:
            continue
            
        clean_tokens.append(token.text)
    
    return " ".join(clean_tokens)

def build_sketch_engine_corpus(csv_path: str, output_dir: str):
    """
    Builds the corpus with metadata-rich filenames for subcorpus contrast[cite: 4].
    """
    print(f"📊 Starting Corpus Construction...")
    df = pd.read_csv(csv_path)
    os.makedirs(output_dir, exist_ok=True)

    start_time = time.time()
    
    for i, row in df.iterrows():
        raw_text = str(row['full_narrative'])
        purified_text = clean_and_mask_text(raw_text)
        
        if len(purified_text.split()) > 10:
            # Metadata-rich naming: Ownership_Region_Country_ID[cite: 3, 4]
            ownership = "Indep" if "independent" in str(row['ownership_type']).lower() else "Chain"
            region = str(row['region']).split()[0].capitalize()
            country = "".join(str(row['country']).title().split())
            h_id = str(row['hotel_id']).upper()
            
            filename = f"{ownership}_{region}_{country}_{h_id}.txt"
            
            with open(os.path.join(output_dir, filename), 'w', encoding='utf-8') as f:
                f.write(purified_text)
                
    print(f"✅ Process completed in {time.time() - start_time:.2f}s.")

def run_audit(output_dir: str):
    """Generates a Top 100 frequency audit to verify purity."""
    from collections import Counter
    words = []
    for f in os.listdir(output_dir):
        with open(os.path.join(output_dir, f), 'r') as file:
            words.extend(file.read().lower().split())
    
    print("\n📝 TOP 100 AUDIT:")
    for i, (w, f) in enumerate(Counter(words).most_common(100), 1):
        print(f"{i}. {w} ({f})")
