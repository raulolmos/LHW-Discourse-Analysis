# config.py

# Regex patterns for UI/UX elements, legal noise, and web functional text
BOILERPLATE_PATTERNS = [
    r'this browser is not supported',
    r'please use a newer browser',
    r'google chrome',
    r'loader bg loading',
    r'scroll down',
    r'official website',
    r'terms and conditions',
    r'privacy policy',
    r'cookie policy',
    r'book now',
    r'all rights reserved',
    r'check availability',
    r'© \d{4}',
    r'read more',
    r'follow us',
    r'copyright',
    r'managed by',
    r'click here',
    r'find out more'
]

# POS (Part-of-Speech) tags to retain for the LHSC (Luxury Hotel Symbolic Capital) analysis
# We focus on Nouns, Adjectives, Verbs, and Adverbs as they carry semantic weight
ALLOWED_POS_TAGS = {'NOUN', 'ADJ', 'VERB', 'ADV'}

# Named Entity Recognition (NER) labels to exclude to avoid brand or geographic bias
# GPE: Countries/Cities, ORG: Companies, PERSON: Names, FAC: Buildings/Hotels
ENTITY_LABELS_TO_EXCLUDE = {'GPE', 'ORG', 'PERSON', 'LOC', 'FAC'}
