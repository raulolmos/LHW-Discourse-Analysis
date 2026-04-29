# processor.py
import pandas as pd
import re
import spacy
from config import BOILERPLATE_PATTERNS, ALLOWED_POS_TAGS, ENTITY_LABELS_TO_EXCLUDE

class CorpusProcessor:
    """
    Core NLP engine to transform raw web narratives into clean, 
    lemmatized tokens for Sketch Engine ingestion.
    """
    def __init__(self, spacy_model="en_core_web_sm"):
        try:
            self.nlp = spacy.load(spacy_model)
        except OSError:
            raise OSError(f"Model {spacy_model} missing. Run: python -m spacy download {spacy_model}")

    def _clean_regex(self, text):
        """Initial noise reduction using regular expressions."""
        if not isinstance(text, str):
            return ""
        
        text = text.lower()
        for pattern in BOILERPLATE_PATTERNS:
            text = re.sub(pattern, '', text)
        
        # Remove URLs and email addresses
        text = re.sub(r'http\S+|www\S+|[\w\.-]+@[\w\.-]+', '', text)
        return text

    def process_text(self, text):
        """
        Executes NLP pipeline:
        1. Boilerplate removal.
        2. Named Entity Filtering (removes specific hotel/city names).
        3. Part-of-Speech Filtering.
        4. Lemmatization.
        """
        raw_clean = self._clean_regex(text)
        if not raw_clean:
            return ""

        doc = self.nlp(raw_clean)
        
        # Identify indices of tokens belonging to excluded named entities
        excluded_indices = []
        for ent in doc.ents:
            if ent.label_ in ENTITY_LABELS_TO_EXCLUDE:
                excluded_indices.extend(range(ent.start, ent.end))

        # Filter tokens based on POS tags and entity exclusion
        cleaned_tokens = [
            token.lemma_.lower() for i, token in enumerate(doc)
            if i not in excluded_indices 
            and not token.is_stop 
            and not token.is_punct 
            and not token.is_digit
            and token.pos_ in ALLOWED_POS_TAGS
        ]

        return " ".join(cleaned_tokens)

    def generate_filename(self, row):
        """
        Creates a standardized filename for metadata-based sub-corpus 
        creation in Sketch Engine. 
        Format: ISO-COUNTRY_OWNERSHIP_HOTEL-ID.txt
        """
        country = str(row['country']).strip().upper()[:3]
        ownership = str(row['ownership_type']).strip().capitalize()
        hotel_id = str(row['hotel_id']).strip().upper()
        return f"{country}_{ownership}_{hotel_id}.txt"
