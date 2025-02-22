from transformers import RobertaForSequenceClassification, RobertaTokenizer

agroot = './agnews/model'

model_name = 'roberta-base'
model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=4)

tokenizer = RobertaTokenizer.from_pretrained(model_name)

tokenizer.save_pretrained(agroot)
model.save_pretrained(agroot)
