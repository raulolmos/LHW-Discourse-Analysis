# main.py
import pandas as pd
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tqdm import tqdm
from src.processor import CorpusProcessor

def run_pipeline():
    # File configuration
    INPUT_FILE = "04_official_sites_text_corpus.csv"
    OUTPUT_FOLDER = "processed_corpus_files"
    
    # Initialize directory
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    print("--- Starting Academic Corpus Processing ---")
    
    # Load dataset
    df = pd.read_csv(INPUT_FILE)
    processor = CorpusProcessor(spacy_model="en_core_web_sm")
    
    processed_log = []

    # Process each row with progress bar
    print(f"Processing {len(df)} hotel narratives...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        # 1. Clean the narrative text
        clean_narrative = processor.process_text(row['full_narrative'])
        
        # 2. Generate systematic filename for Sketch Engine metadata
        filename = processor.generate_filename(row)
        file_path = os.path.join(OUTPUT_FOLDER, filename)

        # 3. Save as individual TXT file
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(clean_narrative)
            
        processed_log.append({
            "filename": filename,
            "original_hotel_id": row['hotel_id'],
            "word_count_processed": len(clean_narrative.split())
        })

    # Export a processing summary for methodology chapter
    log_df = pd.DataFrame(processed_log)
    log_df.to_csv("cleaning_summary_report.csv", index=False)
    
    print(f"\nSUCCESS: {len(log_df)} files exported to '{OUTPUT_FOLDER}'.")
    print("Ready for Sketch Engine ingestion.")

if __name__ == "__main__":
    run_pipeline()
