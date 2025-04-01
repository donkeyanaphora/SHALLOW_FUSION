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
Below is a refined and more visually appealing version of your setup instructions:

---

## Setup

### 1. Install Dependencies

#### System Dependencies

- **Linux:**
  ```bash
  apt-get install lftp
  ```

- **Mac:**
  ```bash
  brew install lftp
  brew install coreutils
  ```

#### Python Dependencies (Cross-Platform)

Install the required Python packages:
```bash
pip install torch transformers tqdm datasets lxml librosa
```

> **Note:** *Librosa is only required for shallow fusion integrations.*


## Download Pubmed XML Files
<!-- ```zsh
mkdir -p pubmed_files
seq 1 100 | xargs -n1 -P 10 -I{} bash -c 'file=$(printf "pubmed25n%04d.xml.gz" {}); curl -o "pubmed_files/$file" "https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/$file"'
``` -->

**Step 1 (Download Data):** 

Donwload all pubmed xml and md5 zip files
```zsh
lftp -c "open ftp://ftp.ncbi.nlm.nih.gov/pubmed/baseline/; mirror --parallel=10 . pubmed_files"
```

<!-- Download all pubmed xml and md5 files in parallel (specify number of files you want valid range is 1-1274):
```zsh
mkdir -p pubmed_files
seq 1 1274 | xargs -n1 -P 10 -I{} bash -c '
  file=$(printf "pubmed25n%04d.xml.gz" {});
  curl -sf -o "pubmed_files/$file" "https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/$file" &&
  curl -sf -o "pubmed_files/$file.md5" "https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/$file.md5"
'
``` -->

**Step 2: (Validate Data)** 

Compare the computed checksum with the expected one. If they match, the filename is logged in valid_files.txt
```zsh
for i in $(seq 1 1274); do
  file=$(printf "pubmed25n%04d.xml.gz" "$i")
  filepath="pubmed_files/$file"
  md5file="$filepath.md5"

  if [ ! -f "$filepath" ] || [ ! -f "$md5file" ]; then
    echo "Skipping $file, or its md5 file, because it does not exist"
    continue
  fi

  if command -v md5sum >/dev/null 2>&1; then
    computed=$(md5sum "$filepath" | cut -d' ' -f1)
  else
    computed=$(md5 "$filepath" | cut -d' ' -f4)
  fi

  expected=$(awk -F'= ' '{print $2}' "$md5file")

  if [ "$computed" = "$expected" ]; then
    echo "$file" >> valid_files.txt
  else
    echo "MD5 mismatch for $file"
  fi
done
```
Downloaded PubMed `.xml.gz` files in `pubmed_files/`.
## Data Pipeline

Run scripts in order:

1. **Extract Abstracts**

```bash
python scripts/01_extract_batches.py
```

2. **Shuffle Data**

```bash
python scripts/02_shuffle_data.py
```
3. **Train Test Split**
```bash
python scripts/03_train_test_split.py
```

4. **Tokenize and batch (train/test)**

```bash
python scripts/04_tokenize_and_batch.py
```

prepared batches (`.pt`) will be in `processed_batches/train/` and `processed_batches/test/`.

## Fine Tuning

- [fine tuning notebook](notebooks/train_model.ipynb)

## Inference + Shallow Fusion
- [inference with fusion notebook](notebooks/train_model.ipynb)
