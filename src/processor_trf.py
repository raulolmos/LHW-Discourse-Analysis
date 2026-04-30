"""
==============================================================================
Module: processor_trf.py
Project: PhD Thesis - Luxury Hospitality Speech Code (LHSC) Validation
Author: [Tu Nombre]
Institution: Universidad Complutense de Madrid (UCM)

Objective: 
Advanced NLP preprocessing for the LHW institutional corpus.
This script utilizes spaCy's Transformer model (en_core_web_trf) to achieve 
high-accuracy POS tagging and lemmatization. It isolates the "brand narrative" 
by aggressively filtering UI/UX noise, geographical/proper nouns (PROPN), 
domain-specific stopwords, and functionally empty "light verbs".
==============================================================================
"""

import os
import re
import spacy
from typing import List

# ==============================================================================
# 1. MODEL INITIALIZATION & CONFIGURATION
# ==============================================================================
# Rationale: We use the Transformer model for context-aware POS tagging.
# NER (Named Entity Recognition) is disabled to optimize processing speed, 
# as proper nouns are handled natively via POS tags (excluding 'PROPN').
try:
    nlp = spacy.load("en_core_web_trf", disable=["ner"])
    print("✅ Loaded Transformer model: en_core_web_trf")
except OSError:
    print("❌ ERROR: Transformer model not found. Run: python -m spacy download en_core_web_trf")
    raise

# ==============================================================================
# 2. THEORETICAL & METHODOLOGICAL DICTIONARIES
# ==============================================================================

# CONCEPT_MAP: Preserves multi-word concepts critical to luxury dimensions (e.g., Savoir-faire)
CONCEPT_MAP = {
    "room service": "room_service",
    "air conditioning": "air_conditioning",
    "front desk": "front_desk",
    "fine dining": "fine_dining",
    "check in": "check_in",
    "check out": "check_out",
    "michelin star": "michelin_star"
}

# DOMAIN_STOPWORDS: Ubiquitous hospitality terms that do not discriminate luxury.
DOMAIN_STOPWORDS = {
    "hotel", "room", "suite", "book", "reservation", "stay", "guest", 
    "lhw", "leading", "world", "price", "click", "website", "property", "resort"
}

# LIGHT_VERBS: Verbs with high functional frequency but low semantic value.
# Filtering these prevents the 'Efficacité' dimension from being artificially inflated.
LIGHT_VERBS = {
    "have", "be", "do", "offer", "provide", "feature", "include", 
    "locate", "make", "take", "get", "find", "use", "require"
}

# ALLOWED_POS: The core syntactic units of the LHSC framework.
# PROPN (Proper Nouns) are strictly EXCLUDED to avoid geographical/brand bias.
ALLOWED_POS = {"NOUN", "ADJ", "VERB"}

# ==============================================================================
# 3. TEXT PURIFICATION PIPELINE
# ==============================================================================

def _clean_ui_noise(text: str) -> str:
    """
    Phase 1: Regex-based removal of boilerplate web artifacts.
    Rationale: Eliminates transactional noise before NLP processing.
    """
    if not text or str(text).lower() == "nan":
        return ""
    
    # Remove HTML, URLs, Emails
    t = re.sub(r'<.*?>', ' ', text)
    t = re.sub(r'http\S+|www\S+|https\S+', '', t, flags=re.MULTILINE)
    t = re.sub(r'\S+@\S+', '', t)
    
    # Custom UI noise patterns observed in the LHW corpus
    ui_patterns = [
        r'skip to (primary )?content', r'check availability', r'book now', 
        r'local time', r'weather \d+.*?c', r'view virtual tour'
    ]
    for pattern in ui_patterns:
        t = re.sub(pattern, ' ', t, flags=re.IGNORECASE)
        
    t = t.lower().strip()
    t = re.sub(r'\s+', ' ', t)
    
    # Apply Concept Fusion
    for raw, fused in CONCEPT_MAP.items():
        t = t.replace(raw, fused)
        
    return t

def is_valid_token(token) -> bool:
    """
    Phase 2: Strict syntactic and semantic filtering of individual tokens.
    Rationale: Isolates tokens that carry the weight of De Barnier's dimensions.
    """
    # 1. Basic sanity checks (alphabetic, not standard stopword, length > 2)
    if not token.is_alpha or token.is_stop or len(token.text) <= 2:
        return False
        
    # 2. POS Tagging filter (Must be Noun, Adjective, or Verb; EXCLUDES Proper Nouns)
    if token.pos_ not in ALLOWED_POS:
        return False
        
    lemma = token.lemma_.lower()
    
    # 3. Domain Stopword filter
    if lemma in DOMAIN_STOPWORDS:
        return False
        
    # 4. Verb Rigor Check (Syntactic & Lexical)
    if token.pos_ == "VERB":
        # Reject if the verb is acting as an auxiliary (e.g., 'has' in 'has been')
        if token.dep_ in ['aux', 'auxpass']:
            return False
        # Reject if it is a 'Light Verb' without luxury dimension value
        if lemma in LIGHT_VERBS:
            return False

    return True

def process_single_narrative(text: str) -> List[str]:
    """
    Processes a single hotel narrative string into a refined list of lemmas.
    """
    clean_text = _clean_ui_noise(text)
    if not clean_text:
        return []

    doc = nlp(clean_text)
    lemmas = [token.lemma_.lower() for token in doc if is_valid_token(token)]
    
    return lemmas

def process_corpus_batch(texts: List[str], batch_size: int = 16) -> List[List[str]]:
    """
    Phase 3: High-performance batch processing for the entire corpus.
    Uses nlp.pipe for optimized Transformer execution.
    """
    cleaned_inputs = [_clean_ui_noise(t) for t in texts]
    results = []
    
    # nlp.pipe yields processed Doc objects efficiently
    for doc in nlp.pipe(cleaned_inputs, batch_size=batch_size):
        lemmas = [token.lemma_.lower() for token in doc if is_valid_token(token)]
        results.append(lemmas)
        
    return results

# ==============================================================================
# MAIN EXECUTION (Example usage for verification)
# ==============================================================================
if __name__ == "__main__":
    sample_text = """
    Skip to primary content. 
    Welcome to the Ritz Paris. The hotel offers an unforgettable fine dining 
    experience. Our dedicated staff has curated bespoke services to indulge 
    your senses. Check availability now!
    """
    print("\n--- LHSC Processor Verification ---")
    print(f"Original Text: {sample_text.strip()}")
    print(f"Processed Lemmas: {process_single_narrative(sample_text)}")
    # Expected output should exclude: Ritz, Paris, hotel, offers, has, check, availability
    # Expected to keep: unforgettable, fine_dining, experience, dedicated, staff, curate, bespoke, service, indulge, sense
