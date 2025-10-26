import pandas as pd
import pickle

def build_vocab(captions_file, min_freq=1):
    word_freq = {}
    with open(captions_file, 'r') as f:
        next(f)  # skip header
        for line in f:
            values = line.strip().split(',')
            if len(values) < 2:
                continue
            caption_text = ",".join(values[1:]).strip()
            for word in caption_text.split():
                word_freq[word] = word_freq.get(word, 0) + 1

    # keep words above min frequency
    vocab = {"<PAD>":0, "<START>":1, "<END>":2, "<UNK>":3}
    idx = 4
    for w, freq in word_freq.items():
        if freq >= min_freq:
            vocab[w] = idx
            idx += 1

    inv_vocab = {i:w for w,i in vocab.items()}

    # save
    with open("vocab.pkl","wb") as f:
        pickle.dump(vocab,f)
    with open("inv_vocab.pkl","wb") as f:
        pickle.dump(inv_vocab,f)

    print(f"Vocabulary size: {len(vocab)}")

if __name__ == "__main__":
    build_vocab("image_captioning_dataset/captions.txt")
