import gzip
import xml.etree.ElementTree as ET
import os 
import json
import glob
from tqdm import tqdm
import shutil
import subprocess

input_dir = "pubmed_files"
os.makedirs("staging", exist_ok=True)
os.makedirs("data", exist_ok=True)

output_jsonl = "staging/pubmed_abstracts.jsonl"
shuffled_output_jsonl = "data/shuffled_pubmed_abstracts.jsonl"

def pubmed_abstract_generator(file_path):
    with gzip.open(file_path, 'rb') as f:
        context = ET.iterparse(f, events=('end',))
        for event, elem in context:
            if elem.tag == 'MedlineCitation':
                abstract = elem.findtext('./Article/Abstract/AbstractText') or ''
                if abstract.strip():
                    yield abstract
                elem.clear()

with open(output_jsonl, 'w', encoding='utf-8') as out_file:
    xml_files = sorted(glob.glob(os.path.join(input_dir, "*.xml.gz")))
    for file in tqdm(xml_files, desc="Extracting abstracts"):
        for abstract_text in pubmed_abstract_generator(file):
            out_file.write(json.dumps({"abstract": abstract_text}) + '\n')

shuf_command = shutil.which('shuf') or shutil.which('gshuf')
if shuf_command is None:
    raise EnvironmentError(
        "Neither 'shuf' nor 'gshuf' is available. Please install GNU coreutils on macOS or ensure shuf is available on Linux."
    )

print("Shuffling abstracts...")
try:
    # Use subprocess.run to execute the shuffling command with error checking.
    result = subprocess.run(
        [shuf_command, output_jsonl, '-o', shuffled_output_jsonl],
        check=True,
        capture_output=True,
        text=True
    )
    print("Shuffled file saved at:", shuffled_output_jsonl)
except subprocess.CalledProcessError as e:
    print("Error during shuffling:")
    print(e.stderr)