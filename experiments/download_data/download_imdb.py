import json
import os
from collections import Counter
from datasets import load_dataset

# Hyperparameters for preprocessing
max_vocab_size = 20000  # Maximum vocabulary size

# Load the IMDB dataset (binary sentiment classification)
dataset = load_dataset("imdb")
train_dataset = dataset["train"]
test_dataset = dataset["test"]

# Tokenize: lowercase and split on whitespace.
def tokenize_function(example):
    tokens = example["text"].lower().split()
    return {"tokens": tokens}

train_dataset = train_dataset.map(tokenize_function, batched=False)
test_dataset = test_dataset.map(tokenize_function, batched=False)

# Build a vocabulary from the training tokens.
counter = Counter()
for tokens in train_dataset["tokens"]:
    counter.update(tokens)

# Reserve special tokens: <pad> for padding and <unk> for unknown words.
vocab = {"<pad>": 0, "<unk>": 1}
for token, _ in counter.most_common(max_vocab_size - len(vocab)):
    vocab[token] = len(vocab)

vocab_save_path = "/home/s2209005/Pytorch_workshop/imdb_data/input/vocab.json"
with open(vocab_save_path, "w") as f:
    json.dump(vocab, f)

# Convert tokens to a bag-of-words (BoW) vector.
def bag_of_words(example):
    bow_vector = [0] * len(vocab)
    for token in example["tokens"]:
        idx = vocab.get(token, vocab["<unk>"])
        bow_vector[idx] += 1
    example["bow"] = bow_vector
    return example

train_dataset = train_dataset.map(bag_of_words, batched=False)
test_dataset = test_dataset.map(bag_of_words, batched=False)

# Set the format to PyTorch tensors (only 'bow' and 'label' are needed).
train_dataset.set_format(type="torch", columns=["bow", "label"])
test_dataset.set_format(type="torch", columns=["bow", "label"])

# Define a custom directory root and subdirectories.
custom_root = "/home/s2209005/Pytorch_workshop/imdb_data/input"
train_save_path = os.path.join(custom_root, "imdb_train")
test_save_path = os.path.join(custom_root, "imdb_test")

# Save the processed datasets to disk.
train_dataset.save_to_disk(train_save_path)
test_dataset.save_to_disk(test_save_path)