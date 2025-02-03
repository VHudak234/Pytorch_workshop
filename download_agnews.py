from datasets import load_dataset
from transformers import RobertaTokenizer

agroot = './agnews'
agnews = load_dataset('ag_news')
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

def tokenize(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)
tokenized_agnews = agnews.map(tokenize, batched=True)

tokenized_agnews.save_to_disk(agroot)