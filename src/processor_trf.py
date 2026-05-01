# ==============================================================================
# PROJECT: LHSC - Luxury Hospitality Speech Code
# PHASE: Corpus Purification & Methodological Validation (Paper 1)
# STATUS: Production Ready / Peer-Review Version
# AUTHOR: LHSC Research Team (UCM)
# ==============================================================================

import os
import re
import time
import pandas as pd
import spacy
import nltk
from nltk.corpus import stopwords
from collections import Counter
from google.colab import drive

# ==============================================================================
# 1. SETUP & ENVIRONMENT INITIALIZATION
# ==============================================================================
def initialize_environment():
    """Mounts Drive and loads the high-precision Transformer model."""
    print("⏳ Initializing environment and loading Transformer-based NLP models...")
    drive.mount('/content/drive', force_remount=True)
    nltk.download('stopwords', quiet=True)
    
    # en_core_web_trf is essential for high-fidelity POS tagging in Luxury narratives
    spacy.prefer_gpu()
    nlp = spacy.load("en_core_web_trf")
    stop_words = set(stopwords.words('english'))
    return nlp, stop_words

# ==============================================================================
# 2. LEXICAL QUALITY CONTROL (THE BLACKLIST)
# ==============================================================================
# This list eliminates technical, transactional, and architectural web noise.
ACADEMIC_BLACKLIST = {
    # UI/UX & Web Artifacts (Noise identified in the audit)
    'html', 'http', 'https', 'home', 'www', 'com', 'org', 'image', 'click',
    'link', 'url', 'menu', 'button', 'icon', 'slider', 'background', 'color',
    'width', 'height', 'padding', 'margin', 'display', 'font', 'search',
    'cookie', 'cooky', 'datum', 'data', 'website', 'page', 'content', 'site',
    'provider', 'browse', 'browser', 'online', 'internet', 'platform', 
    'app', 'download', 'user', 'visitor', 'navigation', 'scroll',
    
    # Transactional & Legal Boilerplate
    'booking', 'reservation', 'preference', 'advertisement', 'policy', 'privacy', 
    'term', 'condition', 'consent', 'manage', 'storage', 'duration', 'necessary', 
    'third', 'party', 'access', 'availability', 'available', 'information',
    
    # Generic Hospitality Noise (Low discriminatory power)
    'stay', 'book', 'room', 'rooms', 'suite', 'suites', 'hotel', 'hotel_id', 
    'include', 'offer', 'offers', 'provide', 'check', 'type', 'number'
}

# ==============================================================================
# 3. CORE PURIFICATION ENGINE
# ==============================================================================
def clean_for_excellence(text, nlp, stop_words):
    """
    Refined purification: Masks entities and extracts semantic substance.
    Targets Nouns and Adjectives to align with De Barnier's luxury dimensions.
    """
    if not isinstance(text, str) or len(text) < 50: return ""
    
    # 3.1. REGEX CLEANING (URLs, Emails, HTML, and Technical Boilerplate)
    t = re.sub(r"https?://\S+|www\.\S+", " ", text)
    t = re.sub(r"\S+@\S+", " ", t)
    t = re.sub(r"<.*?>", " ", t)  # HTML tags
    t = re.sub(r"\{.*?\}", " ", t)  # Inline CSS/JS
    t = re.sub(r"(?i)general manager:?.*?\n|fax:?.*?(\n|$)|code (cin|cir|ciu).*? ", " ", t)
    
    # 3.2. NLP PROCESSING
    doc = nlp(t)
    clean_tokens = []
    
    for token in doc:
        # Avoid brand and geographic bias via NER masking
        if token.ent_type_ in ["GPE", "LOC", "ORG", "PERSON"]: continue 
        
        low_lemma = token.lemma_.lower().strip()
        
        # QUALITY FILTER: Only strictly alphabetic Nouns and Adjectives > 3 chars
        if (low_lemma not in stop_words and 
            low_lemma not in ACADEMIC_BLACKLIST and 
            len(low_lemma) > 3 and 
            token.pos_ in ['NOUN', 'ADJ'] and
            token.is_alpha):
            clean_tokens.append(low_lemma)
            
    return " ".join(clean_tokens)

# ==============================================================================
# 4. CORPUS PIPELINE EXECUTION
# ==============================================================================
def execute_corpus_pipeline(csv_path, output_dir, nlp, stop_words):
    """Processes the CSV and generates purified TXT files for the audit."""
    df = pd.read_csv(csv_path)
    os.makedirs(output_dir, exist_ok=True)
    print(f"📄 Processing {len(df)} hotel narratives...")
    
    start = time.time()
    for _, row in df.iterrows():
        purified = clean_for_excellence(str(row['full_narrative']), nlp, stop_words)
        if len(purified.split()) > 10:
            # Filename metadata for future subgroup analysis (Ownership, Region, ID)
            own = "Indep" if "independent" in str(row['ownership_type']).lower() else "Chain"
            reg = str(row['region']).split()[0].capitalize()
            hid = str(row['hotel_id']).upper()
            
            with open(f"{output_dir}/{own}_{reg}_{hid}.txt", 'w', encoding='utf-8') as f:
                f.write(purified)
    
    print(f"✅ Pipeline completed in {time.time() - start:.2f}s")

# ==============================================================================
# 5. METHODOLOGICAL AUDIT (THE SUBSTANCE CORE)
# ==============================================================================
def run_excellence_audit(output_dir, min_freq=10, min_docs=5):
    """
    Final validation of the LHSC Core.
    Justification: 
    - min_freq=10 ensures statistical stability (Zipf's Law).
    - min_docs=5 ensures cross-institutional institutionalization.
    """
    word_counts = Counter()
    doc_appearance = Counter()
    
    files = [f for f in os.listdir(output_dir) if f.endswith('.txt')]
    total_docs = len(files)

    for f in files:
        with open(os.path.join(output_dir, f), 'r', encoding='utf-8') as file:
            tokens = file.read().split()
            unique_tokens = set(tokens)
            for t in unique_tokens:
                doc_appearance[t] += 1
            for t in tokens:
                word_counts[t] += 1

    substantive_lexicon = [
        (w, word_counts[w], doc_appearance[w]) 
        for w in word_counts 
        if word_counts[w] >= min_freq and doc_appearance[w] >= min_docs
    ]
    
    substantive_lexicon.sort(key=lambda x: x[1], reverse=True)
    
    df_audit = pd.DataFrame(substantive_lexicon, columns=['Target_Word', 'Total_Freq', 'Doc_Count'])
    df_audit['Dispersion_Pct'] = (df_audit['Doc_Count'] / total_docs) * 100
    
    print(f"\n💎 LHSC SUBSTANCE CORE (Stability Threshold: Freq >= {min_freq}, Docs >= {min_docs})")
    print(df_audit.head(50).to_string(index=False))

    return df_audit

# ==============================================================================
# 6. MAIN EXECUTION
# ==============================================================================
# if __name__ == "__main__":
#    nlp_engine, stops = initialize_environment()
#    execute_corpus_pipeline('INPUT_PATH.csv', 'OUTPUT_DIR', nlp_engine, stops)
#    lex_core = run_excellence_audit('OUTPUT_DIR', min_freq=10, min_docs=5)
