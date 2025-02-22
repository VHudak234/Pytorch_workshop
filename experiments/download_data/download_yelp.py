import numpy as np
from datasets import load_dataset
from gensim.models import Word2Vec
import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt')

yelp = load_dataset('yelp_review_full')

def sample(dataset, percentage = 0.15):
    sample_size = int(len(dataset)*percentage)
    return dataset.shuffle(seed=75).select(range(sample_size))

train_sample = sample(yelp['train'])
test_sample = sample(yelp['test'])

train_text = [word_tokenize(text.lower()) for text in train_sample['text']]
test_text = [word_tokenize(text.lower()) for text in test_sample['text']]

embedding_dim = 300
w2v_model = Word2Vec(sentences=train_text, vector_size=embedding_dim, window=5, min_count=2, workers=4)

def get_embedding(text_tokens, model, dim):
    valid_vectors = [model.wv[word] for word in text_tokens if word in model.wv]
    return np.mean(valid_vectors, axis=0) if valid_vectors else np.zeros(dim)

X_train = np.array([get_embedding(text, w2v_model, embedding_dim) for text in train_text])
y_train = np.array(train_sample["label"])
X_test = np.array([get_embedding(text, w2v_model, embedding_dim) for text in test_text])
y_test = np.array(test_sample["label"])

root='/Users/vincehudak/Documents/Intellij Projects/Pytorch_workshop/yelp/input'

np.savez(f'{root}/yelp_data_fnn.npz', X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
w2v_model.save("word2vec.model")