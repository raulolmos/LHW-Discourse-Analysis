"""
==============================================================================
Module: processor_trf.py
Project: PhD Thesis - Luxury Hospitality Speech Code (LHSC) Validation
Institution: Universidad Complutense de Madrid (UCM)

Objective: 
Advanced NLP preprocessing pipeline for the LHW institutional corpus.
This module leverages spaCy's Transformer model (en_core_web_trf) for 
high-accuracy POS tagging and lemmatization. It isolates the brand narrative 
by rigorously filtering UI/UX noise, geographical/brand proper nouns (PROPN), 
domain-specific stopwords, and functionally empty 'light verbs'.

Additionally, it handles the ingestion of the raw CSV dataset and the 
generation of individual clean TXT files optimized for Sketch Engine analysis.
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
# Rationale: Utilizing the Transformer model ensures context-aware tokenization.
# Named Entity Recognition (NER) is disabled to optimize computational overhead,
# as geographic and brand bias is handled natively via POS tag exclusion (PROPN).
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

# DOMAIN_STOPWORDS: Ubiquitous hospitality terms that lack discriminative power 
# for De Barnier et al.'s luxury dimensions.
DOMAIN_STOPWORDS = {
    "hotel", "room", "suite", "book", "reservation", "stay", "guest", 
    "lhw", "leading", "world", "price", "click", "website", "property", "resort"
}

# LIGHT_VERBS: High-frequency functional verbs with negligible semantic weight.
# Exclusion prevents artificial inflation of the 'Efficacité' dimension.
LIGHT_VERBS = {
    "have", "be", "do", "offer", "provide", "feature", "include", 
    "locate", "make", "take", "get", "find", "use", "require"
}

# ALLOWED_POS: The core syntactic units forming the LHSC framework.
# Explicitly excludes PROPN to mitigate geographic and commercial bias.
ALLOWED_POS = {"NOUN", "ADJ", "VERB"}

# ==============================================================================
# 3. TEXT PURIFICATION PIPELINE
# ==============================================================================

def _clean_ui_noise(text: str) -> str:
    """
    Phase 1: Regex-based purge of digital boilerplate and transactional noise.
    """
    if not isinstance(text, str) or text.lower() == "nan" or not text.strip():
        return ""
    
    # Strip HTML tags, URLs, and Email addresses
    t = re.sub(r'<.*?>', ' ', text)
    t = re.sub(r'http\S+|www\S+|https\S+', '', t, flags=re.MULTILINE)
    t = re.sub(r'\S+@\S+', '', t)
    
    # Custom LHW institutional boilerplate patterns
    ui_patterns = [
        r'skip to (primary )?content', r'check availability', r'book now', 
        r'local time', r'weather \d+.*?c', r'view virtual tour'
    ]
    for pattern in ui_patterns:
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
    # Exclude non-alphabetic tokens, standard stopwords, and micro-tokens
    if not token.is_alpha or token.is_stop or len(token.text) <= 2:
        return False
        
    # Restrict to designated Parts of Speech (Noun, Adjective, Verb)
    if token.pos_ not in ALLOWED_POS:
        return False
        
    lemma = token.lemma_.lower()
    
    # Apply Domain Stopword filter
    if lemma in DOMAIN_STOPWORDS:
        return False
        
    # Apply Light Verb and Auxiliary Verb filters
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
    Orchestrates the ingestion of the raw CSV dataset, applies the NLP pipeline,
    and exports purified TXT files ready for Sketch Engine indexing.
    """
    print(f"\n📊 Initializing Sketch Engine Corpus Builder...")
    print(f"Reading dataset from: {csv_filepath}")
    
    try:
        df = pd.read_csv(csv_filepath)
        print(f"📁 Detected {len(df)} institutional narratives.")
    except Exception as e:
        print(f"❌ Critical Error: Unable to read CSV. Details: {e}")
        return

    # Data validation based on the 04_official_sites_text_corpus.csv schema
    id_col = 'hotel_id'
    text_col = 'full_narrative'
    
    if id_col not in df.columns or text_col not in df.columns:
        print(f"❌ Critical Error: Expected columns '{id_col}' and '{text_col}' not found.")
        return

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    start_time = time.time()
    print("\n⚙️ Executing LHSC Transformer Pipeline. This may take a few minutes...")

    # Extract target data
    raw_texts = df[text_col].fillna("").astype(str).tolist()
    file_names = df[id_col].fillna("unknown_hotel").astype(str).tolist()

    # Process through the NLP engine
    processed_corpus = process_corpus_batch(raw_texts, batch_size=16)

    # Export to individual TXT files
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
    print(f"\n✅ Pipeline execution completed successfully in {elapsed_time:.2f} seconds.")
    print(f"📄 Generated {files_created} purified TXT files at: {output_dir}")
    print("   The corpus is now ready for Sketch Engine ingestion and Kappa sampling.")

# Allows CLI execution for advanced automated pipelines
if __name__ == "__main__":
    print("LHSC Processor Module loaded. Import this module to use its functions.")
