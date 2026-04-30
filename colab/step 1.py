# --- STEP 1: CLEAN AND CLONE ---
import os
import shutil

# Remove previous attempts to ensure a clean state
if os.path.exists('/content/LHW-Discourse-Analysis'):
    shutil.rmtree('/content/LHW-Discourse-Analysis')

%cd /content/
!git clone https://github.com/raulolmos/LHW-Discourse-Analysis.git
%cd LHW-Discourse-Analysis

# --- STEP 2: INSTALL DEPENDENCIES ---
!pip install -r requirements.txt
!python -m spacy download en_core_web_sm

# --- STEP 3: RUN PIPELINE ---
# We set the PYTHONPATH to the current directory to help Python find 'src'
!PYTHONPATH=. python main.py
