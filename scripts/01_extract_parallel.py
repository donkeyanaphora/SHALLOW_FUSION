import os
import gzip
import json
from lxml import etree
import multiprocessing as mp
from tqdm import tqdm

# Adjust these paths as needed
INPUT_DIR = "pubmed_files"
VALID_FILELIST = "valid_files.txt"
OUTPUT_JSONL = "staging/pubmed_abstracts.jsonl"

def parse_one_file(file_path):
    """
    Parses a single gzipped PubMed XML file, returning a single string with
    one JSON line per abstract, joined by newline characters.
    """
    lines = []
    with gzip.open(file_path, 'rb') as f:
        # 'tag=MedlineCitation' speeds up lxml by only yielding those elements.
        context = etree.iterparse(f, events=('end',), tag='MedlineCitation')
        for event, elem in context:
            pmid = elem.findtext('./PMID') or ''
            abstract = elem.findtext('./Article/Abstract/AbstractText') or ''
            if abstract.strip():
                # JSON-encode the line. ensure_ascii=False can be faster if you mostly have ASCII data.
                line = json.dumps({"pmid":pmid, "abstract": abstract}, ensure_ascii=False)
                lines.append(line)
            # Clear to free memory
            elem.clear()
    # Return them joined by newline
    return "\n".join(lines)

def parallel_extract(xml_files, output_path):
    """
    Spawns a pool of worker processes. Each process runs parse_one_file on a subset of the XML files.
    The results are concatenated (in the main process) into a single output file.
    """
    # Create the output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Open the output file once in the main process
    with open(output_path, 'w', encoding='utf-8') as out:
        # Use as many processes as you have CPU cores
        with mp.Pool(processes=mp.cpu_count()) as pool:
            # imap() gives results in the same order as input, but doesn't block until all are done
            # Using tqdm for a progress bar (total=len(xml_files)).
            for lines_str in tqdm(pool.imap(parse_one_file, xml_files), total=len(xml_files), desc="Extracting in parallel"):
                if lines_str:
                    # Write the chunk from each file as we get it
                    out.write(lines_str + "\n")

if __name__ == "__main__":
    # Read the list of valid file names
    with open(VALID_FILELIST, "r", encoding="utf-8") as vf:
        valid_files = [line.strip() for line in vf if line.strip()]

    # Convert these to full paths
    xml_files = [os.path.join(INPUT_DIR, filename) for filename in valid_files]
    xml_files.sort()

    # Run the parallel extraction
    parallel_extract(xml_files, OUTPUT_JSONL)

    print(f"Done! Extracted abstracts into: {OUTPUT_JSONL}")
