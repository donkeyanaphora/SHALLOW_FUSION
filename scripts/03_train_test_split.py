import os
import random

shuffled_file = "data/shuffled_pubmed_abstracts.jsonl"
train_file = "data/train.jsonl"
test_file = "data/test.jsonl"

# ratio should be fine for extremely large datasets like pubmed but adjust accordingly
test_ratio = 0.01
random.seed(42)

print("Splitting into train and test:", shuffled_file)
print(f"Using {1 - test_ratio}/{test_ratio} split")

with open(shuffled_file, 'r') as f_in, \
     open(train_file, 'w') as f_train, \
     open(test_file, 'w') as f_test:
    
    for line in f_in:
        if random.random() < test_ratio:
            f_test.write(line)
        else:
            f_train.write(line)

print("Split done.")
print(f"Train file: {train_file}")
print(f"Test file: {test_file}")
