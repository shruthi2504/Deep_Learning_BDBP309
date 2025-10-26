import pickle
import torch
from pandas.io.formats.printing import pprint_thing
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

# ----------------------
# Dataset
# ----------------------
class ImageCaptionDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_feat = torch.tensor(row["features"], dtype=torch.float32)
        caption_tokens = torch.tensor(row["caption_tokens"], dtype=torch.long)
        return img_feat, caption_tokens

def collate_fn(batch, vocab):
    img_feats, captions = zip(*batch)
    img_feats = torch.stack(img_feats, 0)
    captions = pad_sequence(captions, batch_first=True, padding_value=vocab["<PAD>"])
    return img_feats, captions

# ----------------------
# LSTM Model
# ----------------------
class ImageCaptionLSTM(nn.Module):
    def __init__(self, img_feat_size, embed_size, hidden_size, vocab_size, embedding_matrix):
        super().__init__()
        self.embed = nn.Embedding.from_pretrained(embedding_matrix, freeze=False)
        self.hidden_init = nn.Linear(img_feat_size, hidden_size)
        self.cell_init = nn.Linear(img_feat_size, hidden_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, img_feat, captions):
        """
        Forward pass: take image features + full caption sequence.
        """
        batch_size = img_feat.size(0)
        h0 = torch.tanh(self.hidden_init(img_feat)).unsqueeze(0)
        c0 = torch.tanh(self.cell_init(img_feat)).unsqueeze(0)
        hidden = (h0, c0)

        # Feed full caption sequence except last token
        inputs = captions[:, :-1]
        embeds = self.embed(inputs)
        outputs, _ = self.lstm(embeds, hidden)
        logits = self.fc(outputs)
        return logits  # shape: [batch, seq_len-1, vocab_size]

# ----------------------
# Training and Evaluation
# ----------------------
def main():
    # Load vocab and data
    with open("vocab.pkl","rb") as f:
        vocab = pickle.load(f)
    with open("inv_vocab.pkl","rb") as f:
        inv_vocab = pickle.load(f)

    ground_truth = pd.read_pickle("ground_truth.pkl")
    features_df = pd.read_pickle("features_to_rnn.pkl")
    merged_df = pd.merge(ground_truth, features_df, on="image")

    train_df, test_df = train_test_split(merged_df, test_size=0.2, random_state=42)

    batch_size = 32
    train_loader = DataLoader(ImageCaptionDataset(train_df), batch_size=batch_size, shuffle=True,
                              collate_fn=lambda b: collate_fn(b, vocab))
    test_loader = DataLoader(ImageCaptionDataset(test_df), batch_size=batch_size, shuffle=False,
                             collate_fn=lambda b: collate_fn(b, vocab))

    embedding_matrix = np.load("embedding_matrix.npy")
    embedding_matrix = torch.tensor(embedding_matrix, dtype=torch.float32)
    vocab_size, embed_size = embedding_matrix.shape

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ImageCaptionLSTM(img_feat_size=2048, embed_size=embed_size, hidden_size=512,
                             vocab_size=vocab_size, embedding_matrix=embedding_matrix).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab["<PAD>"])

    # ----------------------
    # Training loop
    # ----------------------
    epochs = 5
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for img_feat, caption_tokens in train_loader:
            img_feat, caption_tokens = img_feat.to(device), caption_tokens.to(device)
            optimizer.zero_grad()
            outputs = model(img_feat, captions=caption_tokens)
            loss = criterion(outputs.view(-1, vocab_size), caption_tokens[:,1:].reshape(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")

    # ----------------------
    # Evaluation (autoregressive generation)
    # ----------------------
    model.eval()
    references, hypotheses = [], []
    smooth_fn = SmoothingFunction().method1

    with torch.no_grad():
        for img_feat, caption_tokens in test_loader:
            img_feat = img_feat.to(device)
            batch_size = img_feat.size(0)

            # Initialize hidden and cell states from image features
            h0 = torch.tanh(model.hidden_init(img_feat)).unsqueeze(0)
            c0 = torch.tanh(model.cell_init(img_feat)).unsqueeze(0)
            hidden = (h0, c0)

            # Prepare start token
            start_token_idx = 1  # or vocab["<START>"] if defined
            input_t = torch.full((batch_size,), start_token_idx, dtype=torch.long, device=device)
            input_t = model.embed(input_t).unsqueeze(1)

            max_len = 20
            generated_ids = []

            for _ in range(max_len):
                out, hidden = model.lstm(input_t, hidden)
                logits = model.fc(out.squeeze(1))  # [batch, vocab_size]
                predicted = logits.argmax(dim=-1)  # [batch]
                generated_ids.append(predicted.cpu().numpy())  # collect predicted ids
                # prepare next input
                input_t = model.embed(predicted).unsqueeze(1)

            # Stack predictions to [batch, seq_len]
            pred_ids = np.stack(generated_ids, axis=1)  # shape: [batch, max_len]

            # Convert predictions and ground truth to word lists
            gt_ids = caption_tokens.numpy()
            for b in range(batch_size):
                pred_caption = [
                    inv_vocab.get(int(idx), "<UNK>")
                    for idx in pred_ids[b]
                    if inv_vocab.get(int(idx), "<UNK>") != "<END>"
                ]
                gt_caption = [
                    inv_vocab.get(int(idx), "<UNK>")
                    for idx in gt_ids[b]
                    if inv_vocab.get(int(idx), "<UNK>") != "<END>"
                ]
                hypotheses.append(pred_caption)
                references.append([gt_caption])

    # Compute BLEU
    bleu_score = corpus_bleu(references, hypotheses, smoothing_function=smooth_fn)
    print(f"Corpus BLEU Score: {bleu_score:.4f}")


if __name__ == "__main__":
    main()


