import torch
import json
from transformers import AutoTokenizer
import os

os.makedirs("processed_batches/train", exist_ok=True)
os.makedirs("processed_batches/test", exist_ok=True)

def batch_generator(file_path, tokenizer, batch_size=50, token_limit=1024, return_labels=False):
    
    input_window = []
    mask_window = []
    # We need an extra token for the shifting operation.
    window_size = token_limit * batch_size + 1

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)

            tokenized = tokenizer(data['abstract'])
            input_ids = tokenized['input_ids'] + [tokenizer.eos_token_id]
            attention_mask = tokenized['attention_mask'] + [1]

            input_window.extend(input_ids)
            mask_window.extend(attention_mask)
            
            while len(input_window) >= window_size and len(mask_window) >= window_size:
                batch_input = input_window[:window_size]
                batch_mask = mask_window[:window_size]

                input_window = input_window[window_size:]
                mask_window = mask_window[window_size:]
                
                input_tensor = torch.tensor(batch_input, dtype=torch.long)
                mask_tensor = torch.tensor(batch_mask, dtype=torch.long)
                
                x = input_tensor[:-1].view(batch_size, token_limit)
                attn = mask_tensor[:-1].view(batch_size, token_limit)
                
                if return_labels:
                    y = input_tensor[1:].view(batch_size, token_limit)
                    yield {'input_ids': x, 'attention_mask': attn, 'labels':y},
                else:
                    yield {'input_ids': x, 'attention_mask': attn}

    # throw away singular leftover (should be singular)
    if input_window:
        print("Incomplete batch remaining; dropping remaining tokens.")


train_path = 'data/train.jsonl'
test_path = 'data/test.jsonl'

train_out_dir = "processed_batches/train"
test_out_dir = "processed_batches/test"

tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
token_limit = 1024
batch_size = 16

train_generator = batch_generator(
    file_path=train_path, 
    tokenizer=tokenizer, 
    batch_size=batch_size, 
    token_limit=token_limit, 
    return_labels=False
)

test_generator = batch_generator(
    file_path=test_path, 
    tokenizer=tokenizer, 
    batch_size=batch_size, 
    token_limit=token_limit, 
    return_labels=False
)

print('-'*50 + 'TRAIN' + '-'*50)
for idx, item in enumerate(train_generator):
    out_train_file = f"{train_out_dir}/batch_{idx:04d}.pt"
    torch.save(item, out_train_file)
    print(f"Saved {out_train_file}")

print('-'*50 + 'TEST' + '-'*50)
for idx, item in enumerate(test_generator):
    out_test_file = f"{test_out_dir}/batch_{idx:04d}.pt"
    torch.save(item, out_test_file)
    print(f"Saved {out_test_file}")