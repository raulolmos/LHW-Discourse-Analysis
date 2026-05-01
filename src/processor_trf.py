# ==============================================================================
# PROJECT: LHSC - Luxury Hospitality Speech Code
# PHASE: Corpus Purification & Methodological Validation (Paper 1)
# STATUS: Production Ready / Peer-Review Version
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
print("⏳ Initializing environment and loading NLP models...")
drive.mount('/content/drive', force_remount=True)
nltk.download('stopwords', quiet=True)

# Using the Transformer model for high-fidelity Part-of-Speech tagging and NER
spacy.prefer_gpu()
nlp = spacy.load("en_core_web_trf")
ENG_STOPWORDS = set(stopwords.words('english'))

# ==============================================================================
# 2. LEXICAL QUALITY CONTROL (THE BLACKLIST)
# ==============================================================================
# We establish a rigorous blacklist to filter out technical, transactional, 
# and generic hospitality noise, ensuring only luxury-specific substance remains.
ACADEMIC_BLACKLIST = {
    # UI/UX & Web Artifacts
    'cookie', 'cooky', 'datum', 'data', 'website', 'page', 'content', 'site', 'link', 
    'provider', 'click', 'browse', 'browser', 'online', 'internet', 'platform', 
    'app', 'download', 'user', 'visitor', 'navigation', 'menu', 'button', 'scroll',
    'html', 'http', 'https', 'home', 'www', 'com', 'org', 'image', 'icon', 'slider', 
    'background', 'color', 'width', 'height', 'padding', 'margin', 'display', 'font', 'search',
    
    # Transactional & Legal
    'booking', 'reservation', 'preference', 'advertisement', 'policy', 'privacy', 
    'term', 'condition', 'consent', 'manage', 'storage', 'duration', 'necessary', 
    'third', 'party', 'access', 'availability', 'available', 'information',
    
    # Generic Hospitality (Noise for LHSC Dimensions)
    'stay', 'book', 'room', 'rooms', 'suite', 'suites', 'hotel', 'hotel_id', 
    'include', 'offer', 'offers', 'provide', 'check', 'type', 'number'
}

# ==============================================================================
# 3. CORE PURIFICATION ENGINE
# ==============================================================================
def clean_for_excellence(text):
    """
    Refined corpus purification: Masks Named Entities (NER), removes web artifacts, 
    and extracts the substantive semantic core (Nouns and Adjectives).
    """
    if not isinstance(text, str) or len(text) < 50: return ""
    
    # 3.1. WEB ARTIFACT AND TECHNICAL NOISE CLEANSING (RegEx)
    t = re.sub(r"https?://\S+|www\.\S+", " ", text) # URLs
    t = re.sub(r"\S+@\S+", " ", t)                  # Email addresses
    t = re.sub(r"<.*?>", " ", t)                    # HTML tags
    t = re.sub(r"\{.*?\}", " ", t)                  # Inline CSS/JS artifacts
    t = re.sub(r"/[a-z0-9/_.-]+\.(?:png|jpg|jpeg|gif|svg|pdf|php|html|js|css)", " ", t, flags=re.I) # Server paths
    t = re.sub(r"(?i)general manager:?.*?\n|fax:?.*?(\n|$)|code (cin|cir|ciu).*? ", " ", t) # Legal & Staff boilerplate
    
    # 3.2. NLP PROCESSING VIA SPACY
    doc = nlp(t)
    clean_tokens = []
    
    for token in doc:
        # 3.3. ENTITY MASKING (Bias Prevention)
        if token.ent_type_ in ["GPE", "LOC", "ORG", "PERSON"]: 
            continue 
        
        low_lemma = token.lemma_.lower().strip()
        
        # 3.4. SUBSTANCE QUALITY FILTER (Alignment with De Barnier et al.)
        if (low_lemma not in ENG_STOPWORDS and 
            low_lemma not in ACADEMIC_BLACKLIST and 
            len(low_lemma) > 3 and 
            token.pos_ in ['NOUN', 'ADJ'] and
            token.is_alpha): 
            clean_tokens.append(low_lemma)
            
    return " ".join(clean_tokens)

# ==============================================================================
# 4. CORPUS PIPELINE EXECUTION
# ==============================================================================
def execute_corpus_pipeline(csv_path, output_dir):
    """Iterates through the raw dataset, purifies text, and exports structured .txt files."""
    df = pd.read_csv(csv_path)
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n📄 Starting NLP pipeline for {len(df)} LHW institutional narratives...")
    
    start = time.time()
    for _, row in df.iterrows():
        purified = clean_for_excellence(str(row['full_narrative']))
        if len(purified.split()) > 10: # Ensure document has substantive content left
            # Metadata-Rich Filenames: Ownership_Region_Country_ID
            own = "Indep" if "independent" in str(row['ownership_type']).lower() else "Chain"
            reg = str(row['region']).split()[0].capitalize()
            cou = "".join(str(row['country']).title().split())
            hid = str(row['hotel_id']).upper()
            
            filename = f"{own}_{reg}_{cou}_{hid}.txt"
            with open(os.path.join(output_dir, filename), 'w', encoding='utf-8') as f:
                f.write(purified)
    
    print(f"✅ NLP Processing completed in {time.time() - start:.2f} seconds.")

# ==============================================================================
# 5. METHODOLOGICAL AUDIT: SEMANTIC DISPERSION & FREQUENCY
# ==============================================================================
def run_excellence_audit(output_dir, min_dispersion=0.10):
    """
    Methodological audit: Filters the extracted lexicon based on overall frequency 
    and cross-document dispersion to identify the true institutional speech code.
    """
    word_counts = Counter()
    doc_appearance = Counter()
    
    files = [f for f in os.listdir(output_dir) if f.endswith('.txt')]
    total_docs = len(files)
    min_docs = int(total_docs * min_dispersion)

    print(f"\n🧐 Auditing semantic substance across {total_docs} purified LHW narratives...")

    for f in files:
        with open(os.path.join(output_dir, f), 'r', encoding='utf-8') as file:
            content = file.read()
            tokens = content.split()
            unique_tokens = set(tokens) # For dispersion counting
            
            for t in unique_tokens:
                doc_appearance[t] += 1
            for t in tokens:
                word_counts[t] += 1

    # Filter by dispersion threshold
    substantive_lexicon = [
        (w, word_counts[w], doc_appearance[w]) 
        for w in word_counts 
        if doc_appearance[w] >= min_docs
    ]
    
    substantive_lexicon.sort(key=lambda x: x[1], reverse=True)

    df_audit = pd.DataFrame(substantive_lexicon, columns=['Target_Word', 'Total_Frequency', 'Document_Count'])
    df_audit['Dispersion_Percentage'] = (df_audit['Document_Count'] / total_docs) * 100
    
    print(f"\n💎 LHSC SUBSTANCE CORE (Threshold: Minimum {min_docs} hotels)")
    print("-" * 75)
    print(df_audit.head(30).to_string(index=False)) 
    print("-" * 75)

    return df_audit

# ==============================================================================
# 6. MAIN EXECUTION BLOCK
# ==============================================================================
if __name__ == "__main__":
    # Define Paths (Adjust if necessary for your Google Drive structure)
    INPUT_CSV = '/content/drive/MyDrive/_doctorado/Fase 2 Beatriz Chaves/04_official_sites_text_corpus.csv'
    FINAL_TXT_DIR = '/content/LHSC_TXT_FILES'
    
    # 1. Process and Purify
    execute_corpus_pipeline(INPUT_CSV, FINAL_TXT_DIR)
    
    # 2. Audit and Validate Dispersions (Set to 15% threshold for rigor)
    lex_core_df = run_excellence_audit(FINAL_TXT_DIR, min_dispersion=0.15)
    
    # 3. Export Validated Core for Linguist Inter-coder Reliability ($\kappa$)
    lex_core_df.to_csv('/content/LHSC_Substance_Core_Validated.csv', index=False)
    print("\n📁 Exported 'LHSC_Substance_Core_Validated.csv' for linguistic validation.")
