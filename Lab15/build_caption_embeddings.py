import pandas as pd
import numpy as np
import pickle

GLOVE_PATH = "wiki_giga_2024_50_MFT20_vectors_seed_123_alpha_0.75_eta_0.075_combined.txt"
CAPTIONS_PATH = "image_captioning_dataset/captions.txt"

def load_glove(path):
    embeddings = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            word = parts[0]
            try:
                vector = np.array(parts[1:], dtype=np.float32)
                embeddings[word] = vector
            except ValueError:
                continue
    return embeddings

def main():
    # Load vocab
    with open("vocab.pkl", "rb") as f:
        vocab = pickle.load(f)

    # Load GloVe embeddings
    glove = load_glove(GLOVE_PATH)
    embed_size = len(next(iter(glove.values())))
    print(f"Using embedding size = {embed_size}")

    # Build embedding matrix
    embedding_matrix = np.random.uniform(-0.1,0.1,(len(vocab), embed_size)).astype(np.float32)
    for word, idx in vocab.items():
        vec = glove.get(word)
        if vec is not None and len(vec) == embed_size:
            embedding_matrix[idx] = vec
    np.save("embedding_matrix.npy", embedding_matrix)
    print(f"Saved embedding_matrix.npy with shape {embedding_matrix.shape}")

    # Process captions and tokenize
    df_list = []
    with open(CAPTIONS_PATH, "r") as f:
        next(f)  # skip header
        for line in f:
            values = line.strip().split(',')
            if len(values) < 2:
                continue
            img = values[0].strip()
            caption = ",".join(values[1:]).strip().split()
            caption_tokens = [vocab.get("<START>")]
            caption_tokens += [vocab.get(w, vocab.get("<UNK>")) for w in caption]
            caption_tokens.append(vocab.get("<END>"))
            df_list.append({"image": img, "caption_tokens": caption_tokens})

    # Save ground truth
    pd.DataFrame(df_list).to_pickle("ground_truth.pkl")
    print(f"Saved ground_truth.pkl with {len(df_list)} captions")

if __name__ == "__main__":
    main()
