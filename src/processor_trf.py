# ==============================================================================
# PROJECT: LHSC - Luxury Hospitality Speech Code
# PHASE: Corpus Purification & Validation (Paper 1)
# STATUS: Production Ready / GitHub Version
# ==============================================================================

import os, re, time, pandas as pd, spacy, torch, nltk, shutil
from nltk.corpus import stopwords
from collections import Counter
from google.colab import drive, files

# 1. SETUP & DRIVE MOUNTING
drive.mount('/content/drive', force_remount=True)
nltk.download('stopwords', quiet=True)
spacy.prefer_gpu()
nlp = spacy.load("en_core_web_trf")
ENG_STOPWORDS = set(stopwords.words('english'))

# 2. DEFINICIÓN DEL "FILTRO DE SUSTANCIA" (ACTUALIZABLE)
# Aquí es donde usted añade el ruido residual que detectamos en la auditoría
ACADEMIC_BLACKLIST = {
    # UI/UX & Web Artifacts
    'cookie', 'cooky', 'datum', 'data', 'website', 'page', 'content', 'site', 'link', 
    'provider', 'click', 'browse', 'browser', 'online', 'internet', 'platform', 
    'app', 'download', 'user', 'visitor', 'navigation', 'menu', 'button', 'scroll',
    
    # Transactional & Legal
    'booking', 'reservation', 'preference', 'advertisement', 'policy', 'privacy', 
    'term', 'condition', 'consent', 'manage', 'storage', 'duration', 'necessary', 
    'third', 'party', 'access', 'availability', 'available', 'information',
    
    # Generic Hospitality (Noise for LHSC Dimensions)
    'stay', 'book', 'room', 'rooms', 'suite', 'suites', 'hotel', 'hotel_id', 
    'include', 'offer', 'offers', 'provide', 'check', 'type', 'number'
}

def clean_for_excellence(text):
    """Refined purification: Masks entities and extracts semantic substance."""
    if not isinstance(text, str) or len(text) < 30: return ""
    
    # regex cleaning (Faxes, Managers, IDs, Emails, URLs)
    t = re.sub(r"(?i)general manager:?.*?\n|fax:?.*?(\n|$)|code (cin|cir|ciu).*? |https?://\S+|www\.\S+|\S+@\S+", " ", text)
    
    doc = nlp(t)
    clean_tokens = []
    for token in doc:
        # NER Masking: Avoid geographic/brand bias but preserve syntax for future analysis
        if token.ent_type_ in ["GPE", "LOC", "ORG"]: continue 
        
        low_lemma = token.lemma_.lower()
        
        # QUALITY FILTER: Substance Only (Nouns & Adjectives)
        if (low_lemma not in ENG_STOPWORDS and 
            low_lemma not in ACADEMIC_BLACKLIST and 
            len(low_lemma) > 3 and 
            token.pos_ in ['NOUN', 'ADJ']):
            clean_tokens.append(low_lemma)
            
    return " ".join(clean_tokens)

# 3. CORE EXECUTION ENGINE
def execute_corpus_pipeline(csv_path, output_dir):
    df = pd.read_csv(csv_path)
    os.makedirs(output_dir, exist_ok=True)
    print(f"📄 Iniciando procesamiento de {len(df)} hoteles...")
    
    start = time.time()
    for _, row in df.iterrows():
        purified = clean_for_excellence(str(row['full_narrative']))
        if len(purified.split()) > 10:
            # Metadata-Rich Filenames: Ownership_Region_Country_ID
            own = "Indep" if "independent" in str(row['ownership_type']).lower() else "Chain"
            reg = str(row['region']).split()[0].capitalize()
            cou = "".join(str(row['country']).title().split())
            hid = str(row['hotel_id']).upper()
            
            with open(f"{output_dir}/{own}_{reg}_{cou}_{hid}.txt", 'w', encoding='utf-8') as f:
                f.write(purified)
    
    print(f"✅ Proceso finalizado en {time.time() - start:.2f}s")

# 4. PATHS & RUN
INPUT_CSV = '/content/drive/MyDrive/_doctorado/Fase 2 Beatriz Chaves/04_official_sites_text_corpus.csv'
FINAL_TXT_DIR = '/content/LHSC_TXT_FILES'

execute_corpus_pipeline(INPUT_CSV, FINAL_TXT_DIR)
