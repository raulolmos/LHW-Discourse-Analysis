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
    Orchestrates the ingestion, application of LHSC NLP pipeline,
    and exports purified TXT files ready for Sketch Engine indexing.
    """
    print(f"\n📊 Initializing LHSC v2.0 Sketch Engine Corpus Builder...")
    
    try:
        df = pd.read_csv(csv_filepath)
        print(f"📁 Detected {len(df)} institutional narratives.")
    except Exception as e:
        print(f"❌ Critical Error: Unable to read CSV. Details: {e}")
        return

    id_col = 'hotel_id'
    text_col = 'full_narrative'
    
    os.makedirs(output_dir, exist_ok=True)

    start_time = time.time()
    print("\n⚙️ Executing Deep UI Purge and Transformer Pipeline...")

    raw_texts = df[text_col].fillna("").astype(str).tolist()
    file_names = df[id_col].fillna("unknown_hotel").astype(str).tolist()

    processed_corpus = process_corpus_batch(raw_texts, batch_size=16)

    files_created = 0
    for name, lemmas in zip(file_names, processed_corpus):
        if len(lemmas) > 0:
            final_text = " ".join(lemmas)
            safe_name = "".join(c for c in name if c.isalnum() or c in ('_', '-')).strip()
            filepath = os.path.join(output_dir, f"{safe_name}.txt")
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(final_text)
                files_created += 1

    elapsed_time = time.time() - start_time
    print(f"\n✅ Execution completed in {elapsed_time:.2f} seconds.")
    print(f"📄 Generated {files_created} absolutely purified TXT files at: {output_dir}")

if __name__ == "__main__":
    print("LHSC Processor Module loaded.")
