import gzip
import xml.etree.ElementTree as ET
import os 
import json
import glob
from tqdm import tqdm

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

print("Shuffling abstracts...")
os.system(f'gshuf {output_jsonl} -o {shuffled_output_jsonl}')
print("Shuffled file saved at:", shuffled_output_jsonl)
