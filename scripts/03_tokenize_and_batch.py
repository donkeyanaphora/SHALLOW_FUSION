import torch
import json
from transformers import AutoTokenizer
import os

os.makedirs("processed_batches/train", exist_ok=True)
os.makedirs("processed_batches/test", exist_ok=True)

def abstract_batch_generator(jsonl_file, batch_size):
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        batch = []
        for line in f:
            record = json.loads(line)
            batch.append(record["abstract"])
            if len(batch) == batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

def save_torch_batch(input_ids, attention_masks, labels, filename):
    torch.save({
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long)
    }, filename)

def tokenize_batch(tokenizer, token_limit, file_in, output_dir, batch_size=1000):
    data_stream = abstract_batch_generator(file_in, batch_size)
    eos_token_id = tokenizer.eos_token_id

    for batch_idx, batch in enumerate(data_stream):
        # as stupid as this feels since we have to iterate over them again to end EOS_TOKEN 
        # it is faster to bath tokenize and then loop...
        tokenized = tokenizer(batch, add_special_tokens=False, padding=False, truncation=False)
        chunked_ids, chunked_attention = [], []
        buffer = []

        # add EOS_ID to end of text this will act as separator when we aggregate abstracts to token limit
        for input_ids in tokenized["input_ids"]:
            # Add EOS token as a delimiter
            input_ids.append(eos_token_id)
            buffer.extend(input_ids)

            # extract valid chunk segment of len token_limit
            while len(buffer) >= token_limit:
                segment = buffer[:token_limit]
                buffer = buffer[token_limit:]
                chunked_ids.append(segment)
                attention_mask = [1]*token_limit
                chunked_attention.append(attention_mask)

        # handle leftover text (this should ensure we only have one observation < token_limit)
        if buffer:
            pad_len = token_limit - len(buffer)
            segment = buffer + [eos_token_id]*pad_len
            chunked_ids.append(segment)
            attention_mask = [1]*len(buffer) + [0]*pad_len
            chunked_attention.append(attention_mask)
            buffer = []

        # convert to PT tensors & save batches as .pt files
        out_file = f"{output_dir}/batch_{batch_idx:04d}.pt"
        save_torch_batch(chunked_ids, chunked_attention, chunked_ids, out_file)
        print(f"Saved {out_file}")


tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
token_limit = 1024

train_path = 'data/train.jsonl'
test_path = 'data/test.jsonl'

train_out = "processed_batches/train"
test_out = "processed_batches/test"

# Run separately for train
print('-'*50 + 'TRAIN' + '-'*50)
tokenize_batch(
    tokenizer=tokenizer, 
    token_limit=token_limit,
    file_in=train_path, 
    output_dir=train_out
    )

# and for test sets
print('-'*50 + 'TEST' + '-'*50)
tokenize_batch(
    tokenizer=tokenizer, 
    token_limit=token_limit,
    file_in=test_path, 
    output_dir=test_out
    )
print('-'*50 + 'COMPLETED' + '-'*50)
print(f'files saved to:\n{train_out}\n{test_out}')