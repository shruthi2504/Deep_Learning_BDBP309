import pickle
import torch
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
# RNN Model
# ----------------------
class ImageCaptionRNN(nn.Module):
    def __init__(self, img_feat_size, embed_size, hidden_size, vocab_size, embedding_matrix):
        super().__init__()
        self.embed = nn.Embedding.from_pretrained(embedding_matrix, freeze=False)
        self.hidden_init = nn.Linear(img_feat_size, hidden_size)
        self.rnn = nn.RNN(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.start_token_idx = 1
        self.hidden_size = hidden_size

    def forward(self, img_feat, max_len=20):
        batch_size = img_feat.size(0)
        h0 = torch.tanh(self.hidden_init(img_feat)).unsqueeze(0)
        start_idx = torch.full((batch_size,), self.start_token_idx, dtype=torch.long, device=img_feat.device)
        input_t = self.embed(start_idx).unsqueeze(1)

        outputs, hidden = [], h0
        for _ in range(max_len):
            out, hidden = self.rnn(input_t, hidden)
            logits = self.fc(out.squeeze(1))
            outputs.append(logits)
            predicted_token = logits.argmax(dim=-1)
            input_t = self.embed(predicted_token).unsqueeze(1)

        return torch.stack(outputs, dim=1)

# ----------------------
# Training
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
    train_loader = DataLoader(ImageCaptionDataset(train_df), batch_size=16, shuffle=True,
                              collate_fn=lambda b: collate_fn(b, vocab))
    test_loader = DataLoader(ImageCaptionDataset(test_df), batch_size=16, shuffle=False,
                             collate_fn=lambda b: collate_fn(b, vocab))

    embedding_matrix = np.load("embedding_matrix.npy")
    embedding_matrix = torch.tensor(embedding_matrix, dtype=torch.float32)
    vocab_size, embed_size = embedding_matrix.shape

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ImageCaptionRNN(img_feat_size=2048, embed_size=embed_size, hidden_size=512,
                             vocab_size=vocab_size, embedding_matrix=embedding_matrix).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab["<PAD>"])

    for epoch in range(5):
        model.train()
        total_loss = 0
        for img_feat, caption_tokens in train_loader:
            img_feat, caption_tokens = img_feat.to(device), caption_tokens.to(device)
            optimizer.zero_grad()
            outputs = model(img_feat, max_len=caption_tokens.size(1))
            loss = sum(criterion(outputs[:, t, :], caption_tokens[:, t]) for t in range(caption_tokens.size(1)))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/5, Loss: {total_loss/len(train_loader):.4f}")

    # Evaluation
    model.eval()
    references, hypotheses = [], []
    smooth_fn = SmoothingFunction().method1
    with torch.no_grad():
        for img_feat, caption_tokens in test_loader:
            img_feat, caption_tokens = img_feat.to(device), caption_tokens.to(device)
            outputs = model(img_feat, max_len=caption_tokens.size(1))
            pred_ids = outputs.argmax(dim=-1).cpu().numpy()
            gt_ids = caption_tokens.cpu().numpy()
            for b in range(img_feat.size(0)):
                pred_caption = [inv_vocab.get(idx,"<UNK>") for idx in pred_ids[b] if inv_vocab.get(idx,"<UNK>") != "<END>"]
                gt_caption = [inv_vocab.get(idx,"<UNK>") for idx in gt_ids[b] if inv_vocab.get(idx,"<UNK>") != "<END>"]
                hypotheses.append(pred_caption)
                references.append([gt_caption])
    bleu_score = corpus_bleu(references, hypotheses, smoothing_function=smooth_fn)
    print(f"Corpus BLEU Score: {bleu_score:.4f}")

if __name__ == "__main__":
    main()
