"""
==============================================================================
Module: processor_trf.py
Project: PhD Thesis - Luxury Hospitality Speech Code (LHSC) Validation
Institution: Universidad Complutense de Madrid (UCM)

Objective: 
Advanced NLP preprocessing pipeline for the LHW institutional corpus.
This module leverages spaCy's Transformer model (en_core_web_trf). 
VERSION 2.0 UPDATE: Integrates deep UI/UX noise purging and transaction 
artifact removal to ensure the corpus strictly reflects brand narrative 
(De Barnier et al. luxury dimensions) and not web interface mechanics.
==============================================================================
"""

import os
import re
import time
import pandas as pd
import spacy
from typing import List

# ==============================================================================
# 1. MODEL INITIALIZATION
# ==============================================================================
try:
    nlp = spacy.load("en_core_web_trf", disable=["ner"])
    print("✅ System Ready: Loaded Transformer model (en_core_web_trf)")
except OSError:
    print("❌ ERROR: Transformer model missing. Execute: python -m spacy download en_core_web_trf")
    raise

# ==============================================================================
# 2. LINGUISTIC & THEORETICAL DICTIONARIES
# ==============================================================================

# CONCEPT_MAP: Preserves multi-word noun phrases critical to luxury dimensions.
CONCEPT_MAP = {
    "room service": "room_service",
    "air conditioning": "air_conditioning",
    "front desk": "front_desk",
    "fine dining": "fine_dining",
    "check in": "check_in",
    "check out": "check_out",
    "michelin star": "michelin_star"
}

# DOMAIN_STOPWORDS: Ubiquitous hospitality terms and UI/Navigational noise.
# Added 'login', 'signup', 'discover', 'view', 'more' to eliminate UI artifacts.
DOMAIN_STOPWORDS = {
    "hotel", "room", "suite", "book", "reservation", "stay", "guest", 
    "lhw", "leading", "world", "price", "click", "website", "property", "resort",
    "login", "signup", "sign", "up", "discover", "more", "view", "menu", "availability",
    "online", "account", "join"
}

# LIGHT_VERBS: High-frequency functional verbs with negligible semantic weight.
LIGHT_VERBS = {
    "have", "be", "do", "offer", "provide", "feature", "include", 
    "locate", "make", "take", "get", "find", "use", "require"
}

ALLOWED_POS = {"NOUN", "ADJ", "VERB"}

# ==============================================================================
# 3. TEXT PURIFICATION PIPELINE (THEORY-DRIVEN REGEX)
# ==============================================================================

def _clean_ui_noise(text: str) -> str:
    """
    Phase 1: Regex-based purge of digital boilerplate, transactional noise,
    and browser compatibility artifacts.
    """
    if not isinstance(text, str) or text.lower() == "nan" or not text.strip():
        return ""
    
    # 1. Strip HTML tags, URLs, and Email addresses
    t = re.sub(r'<.*?>', ' ', text)
    t = re.sub(r'http\S+|www\S+|https\S+', '', t, flags=re.MULTILINE)
    t = re.sub(r'\S+@\S+', '', t)
    
    # 2. Target specific browser/technical noise
    browser_noise = [
        r'official browser support.*?browser', 
        r'this browser is not supported',
        r'please use a newer browser',
        r'loader bg loading',
        r'skip to (primary )?content'
    ]
    for pattern in browser_noise:
        t = re.sub(pattern, ' ', t, flags=re.IGNORECASE)
        
    # 3. Target common transactional/administrative noise
    transactional_noise = [
        r'rate guarantee', r'general manager', r'code cin', r'code fax',
        r'cir code', r'ciu code', r'codice cir', r'codice cin', r'fax \+?\d+',
        r'manage your booking', r'subscribe to our newsletter',
        r'check availability', r'book now', r'local time', r'weather \d+.*?c',
        r'view virtual tour'
    ]
    for pattern in transactional_noise:
        t = re.sub(pattern, ' ', t, flags=re.IGNORECASE)
        
    t = t.lower().strip()
    t = re.sub(r'\s+', ' ', t)
    
    # Execute Concept Fusion
    for raw, fused in CONCEPT_MAP.items():
        t = t.replace(raw, fused)
        
    return t

def is_valid_token(token) -> bool:
    """
    Phase 2: Syntactic and semantic token evaluation based on LHSC parameters.
    """
    if not token.is_alpha or token.is_stop or len(token.text) <= 2:
        return False
        
    if token.pos_ not in ALLOWED_POS:
        return False
        
    lemma = token.lemma_.lower()
    
    if lemma in DOMAIN_STOPWORDS:
        return False
        
    if token.pos_ == "VERB":
        if token.dep_ in ['aux', 'auxpass']:
            return False
        if lemma in LIGHT_VERBS:
            return False

    return True

def process_corpus_batch(texts: List[str], batch_size: int = 16) -> List[List[str]]:
    """
    Phase 3: High-performance batch processing utilizing the Transformer pipeline.
    """
    cleaned_inputs = [_clean_ui_noise(t) for t in texts]
    results = []
    
    for doc in nlp.pipe(cleaned_inputs, batch_size=batch_size):
        lemmas = [token.lemma_.lower() for token in doc if is_valid_token(token)]
        results.append(lemmas)
        
    return results

# ==============================================================================
# 4. SKETCH ENGINE CORPUS COMPILER
# ==============================================================================

def build_sketch_engine_corpus(csv_filepath: str, output_dir: str):
    """
    Orchestrates the ingestion, applies the LHSC NLP pipeline,
    and exports purified TXT files with metadata-rich filenames 
    optimized for Sketch Engine subcorpus creation.
    """
    print(f"\n📊 Initializing LHSC v2.1 Sketch Engine Corpus Builder...")
    
    try:
        # We read the full matrix to access categorical metadata
        df = pd.read_csv(csv_filepath)
        print(f"📁 Detected {len(df)} institutional narratives.")
    except Exception as e:
        print(f"❌ Critical Error: Unable to read CSV. Details: {e}")
        return

    # Define paths
    os.makedirs(output_dir, exist_ok=True)

    start_time = time.time()
    print("\n⚙️ Executing Deep UI Purge and Transformer Pipeline...")

    # NLP batch processing (The 'Brain' of the process)
    raw_texts = df['full_narrative'].fillna("").astype(str).tolist()
    processed_corpus = process_corpus_batch(raw_texts, batch_size=16)

    # PHASE 4: METADATA-RICH EXPORT[cite: 1, 3, 4]
    # Rationale: Incorporating 'ownership', 'region', and 'country' into the filename 
    # allows Sketch Engine to automatically generate subcorpora for contrastive analysis.
    files_created = 0
    
    for i, row in df.iterrows():
        lemmas = processed_corpus[i]
        
        if len(lemmas) > 0:
            # 1. Sanitize and simplify metadata for filenames
            # Independent -> Indep | Corporate/Chain -> Chain
            ownership = "Indep" if "independent" in str(row['ownership_type']).lower() else "Chain"
            
            # Region (first word only to avoid spaces: e.g., 'Western Europe' -> 'Western')
            region = str(row['region']).split()[0].capitalize()
            
            # Country (Remove spaces: e.g., 'United Kingdom' -> 'Unitedkingdom')
            country = "".join(str(row['country']).title().split())
            
            # ID (Unique identifier)
            h_id = str(row['hotel_id']).upper()

            # 2. Construct the "Scientific Name" of the file
            # Format: OWNERSHIP_REGION_COUNTRY_ID.txt
            rich_filename = f"{ownership}_{region}_{country}_{h_id}.txt"
            
            # 3. Save to disk
            filepath = os.path.join(output_dir, rich_filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(" ".join(lemmas))
            
            files_created += 1

    elapsed_time = time.time() - start_time
    print(f"\n✅ Execution completed in {elapsed_time:.2f} seconds.")
    print(f"📄 Generated {files_created} metadata-tagged files at: {output_dir}")
    print("   Ready for Sketch Engine 'Expert Mode' ingestion.")

if __name__ == "__main__":
    print("LHSC Processor Module loaded.")
