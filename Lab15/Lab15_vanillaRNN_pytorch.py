# from torchtext.vocab import build_vocab_from_iterator
# from torchtext.data.utils import get_tokenizer
#
# tokenizer = get_tokenizer('basic_english')
#
# captions = [
#     "A girl going into a wooden building .",
#     "A little girl climbing into a wooden playhouse ."
# ]
#
# # Yielding tokens for all captions
# def yield_tokens(captions):
#     for caption in captions:
#         yield tokenizer(caption)
#
# # Build one vocabulary/dictionary from all captions
# vocab = build_vocab_from_iterator(yield_tokens(captions))
#
#
# for c in captions:
#     tokens = tokenizer(c)
#     indices = [vocab[t] for t in tokens]
#     print("Tokens:", tokens)
#     print("Indices:", indices)
#



import torch
import torch.nn as nn
import numpy as np
from torchtext.vocab import GloVe

captions = [
    "a cat sitting on the mat",
    "a dog running in the park"
]

words = set(word for cap in captions for word in cap.lower().split())
word2idx = {word: idx for idx, word in enumerate(words)}
idx2word = {idx: word for word, idx in word2idx.items()}
vocab_size = len(word2idx)

print("Vocabulary:", word2idx)

embed_dim = 5
glove = GloVe(name='6B', dim=embed_dim)

weights_matrix = np.zeros((vocab_size, embed_dim))
for word, idx in word2idx.items():
    if word in glove.stoi:            # check if GloVe has this word
        weights_matrix[idx] = glove[word].numpy()  # assign pretrained vector
    else:
        weights_matrix[idx] = np.random.normal(scale=0.6, size=(embed_dim,))  # random init
weights_tensor = torch.tensor(weights_matrix, dtype=torch.float)
embedding_layer = nn.Embedding.from_pretrained(weights_tensor, freeze=False)
caption = "a cat sitting on the mat".split()
indices = torch.tensor([word2idx[w] for w in caption])
embedded = embedding_layer(indices)  # shape: [seq_len, embed_dim]

print(embedded)


