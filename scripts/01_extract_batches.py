import os
import gzip
import json
import logging
from lxml import etree
from tqdm import tqdm
import time

# Adjust these paths as needed
INPUT_DIR = "pubmed_files"
VALID_FILELIST = "valid_files.txt"
OUTPUT_JSONL = "staging/pubmed_abstracts.jsonl"
BATCH_SIZE = 10  # Number of files to process in one batch
BATCH_DELAY = 1  # Seconds to wait between batches

# Set up logging
logging.basicConfig(filename='error.log', level=logging.ERROR, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

def parse_one_file(file_path):
    """
    Parses a single gzipped PubMed XML file, returning a single string with
    one JSON line per abstract, joined by newline characters.
    """
    lines = []
    try:
        with gzip.open(file_path, 'rb') as f:
            # 'tag=MedlineCitation' speeds up lxml by only yielding those elements.
            context = etree.iterparse(f, events=('end',), tag='MedlineCitation')
            for event, elem in context:
                pmid = elem.findtext('./PMID') or ''
                abstract = elem.findtext('./Article/Abstract/AbstractText') or ''
                if abstract.strip():
                    # JSON-encode the line.
                    line = json.dumps({"pmid": pmid, "abstract": abstract}, ensure_ascii=False)
                    lines.append(line)
                # Clear the element to free memory
                elem.clear()
    except Exception as e:
        logging.error(f"Error processing {file_path}: {e}")
        return ""
    return "\n".join(lines)

def batch_sequential_extract(xml_files, output_path, batch_size=BATCH_SIZE, delay=BATCH_DELAY):
    """
    Processes files in batches sequentially. After each batch, the output is flushed,
    and a short delay is introduced to give the system a break.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as out:
        total_files = len(xml_files)
        for batch_start in range(0, total_files, batch_size):
            batch_files = xml_files[batch_start:batch_start+batch_size]
            # Process the current batch sequentially.
            for file_path in tqdm(batch_files, desc=f"Processing batch {batch_start//batch_size+1}", leave=False):
                result = parse_one_file(file_path)
                if result:
                    out.write(result + "\n")
            out.flush()  # Ensure the output is written to disk after each batch.
            # Optionally, add a delay between batches to reduce continuous load.
            time.sleep(delay)
            print(f"Completed batch {batch_start//batch_size+1}/{(total_files-1)//batch_size+1}")
    
if __name__ == "__main__":
    # Read the list of valid file names.
    with open(VALID_FILELIST, "r", encoding="utf-8") as vf:
        valid_files = [line.strip() for line in vf if line.strip()]

    # Convert these to full paths and sort them.
    xml_files = [os.path.join(INPUT_DIR, filename) for filename in valid_files]
    xml_files.sort()

    batch_sequential_extract(xml_files, OUTPUT_JSONL)
    print(f"Done! Extracted abstracts into: {OUTPUT_JSONL}")
