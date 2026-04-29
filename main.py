# main.py
import os
import sys
import pandas as pd
from tqdm import tqdm

# Add the project root to the Python path to ensure 'src' is discoverable
# This solves the ModuleNotFoundError in Colab/Linux environments
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)

# Import our custom package components
from src.processor import CorpusProcessor

def run_pipeline():
    # File configuration
    INPUT_FILE = os.path.join(script_dir, "data", "raw", "04_official_sites_text_corpus.csv")
    OUTPUT_FOLDER = os.path.join(script_dir, "data", "processed", "corpus_files")
    
    # Initialize output directory
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    print("--- Starting Academic Corpus Processing ---")
    
    # Check if input file exists before processing
    if not os.path.exists(INPUT_FILE):
        print(f"ERROR: Input file not found at {INPUT_FILE}")
        return

    # Load dataset
    df = pd.read_csv(INPUT_FILE)
    processor = CorpusProcessor(spacy_model="en_core_web_sm")
    
    processed_log = []

    print(f"Processing {len(df)} hotel narratives...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        # 1. Clean the narrative text
        # Using 'full_narrative' as the primary source of discourse
        clean_text = processor.process_text(row['full_narrative'])
        
        # 2. Generate systematic filename for Sketch Engine metadata
        filename = processor.generate_filename(row)
        file_path = os.path.join(OUTPUT_FOLDER, filename)

        # 3. Save as individual TXT file
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(clean_text)
            
        processed_log.append({
            "filename": filename,
            "original_hotel_id": row['hotel_id'],
            "word_count_processed": len(clean_text.split())
        })

    # Export a processing summary for methodology chapter
    log_df = pd.DataFrame(processed_log)
    log_df.to_csv(os.path.join(script_dir, "data", "processed", "cleaning_summary.csv"), index=False)
    
    print(f"\nSUCCESS: {len(log_df)} files exported to '{OUTPUT_FOLDER}'.")

if __name__ == "__main__":
    run_pipeline()
