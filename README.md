# PubMed Abstracts LM Pipeline

Simple, reproducible pipeline for preparing PubMed abstracts for language modeling, later integrated with shallow fusion for downstream ASR tasks.  
Read about the motivation [**✨👉 HERE 👈✨**](ARTICLE.md)

## Project Structure

```
project_root/
├── pubmed_files/        # PubMed XML files (.xml.gz)
├── staging/             # Intermediate JSONL files
├── data/                # train/test datasets (JSONL)
├── processed_batches/   # tokenized batches for train/test (.pt)
├── scripts/             # Data processing scripts
└── notebooks/           # Notebooks for training and inference
```

## Setup

Install dependencies:

```bash
pip install torch transformers tqdm datasets librosa
brew install coreutils  # for gshuf (macOS)
```
Librosa is only for shallow fusion integrations. 

## Download Pubmed XML Files
**Run:**
```zsh
mkdir -p pubmed_files
# {1..N}
for i in {1..5}; do
    file=$(printf "pubmed25n%04d.xml.gz" "$i")
    curl -o "pubmed_files/$file" "https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/$file"
done
```
Downloaded PubMed `.xml.gz` files in `pubmed_files/`.
## Data Pipeline

Run scripts in order:

1. **Extract and shuffle abstracts**

```bash
python scripts/01_load_and_shuffle.py
```

2. **Split train/test sets**

```bash
python scripts/02_split_train_test.py
```

3. **Tokenize and batch (train/test)**

```bash
python scripts/03_tokenize_and_batch.py
```

prepared batches (`.pt`) will be in `processed_batches/train/` and `processed_batches/test/`.

## Fine Tuning

- [fine tuning notebook](notebooks/train_model.ipynb)

## Inference + Shallow Fusion
- [inference with fusion notebook](notebooks/train_model.ipynb)
