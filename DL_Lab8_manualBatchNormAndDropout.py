import torch
from torch import nn
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import numpy as np

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

# ---------- Your manual BatchNorm ----------
def batch_stats(x):
    mean = np.mean(x, axis=0)
    var = np.var(x, axis=0)
    return mean, var

def normalize(x, mean, var, eps=1e-5):
    return (x - mean) / np.sqrt(var + eps)

def affine_transform(x_hat, gamma, beta):
    return (x_hat * gamma) + beta

def batchnorm_forward(x, gamma, beta, eps=1e-5):
    x_np = x.detach().cpu().numpy()
    gamma_np = gamma.detach().cpu().numpy()
    beta_np  = beta.detach().cpu().numpy()
    mean, var = batch_stats(x_np)
    x_hat = normalize(x_np, mean, var, eps)
    out = affine_transform(x_hat, gamma_np, beta_np)
    return torch.from_numpy(out).to(x.device).float()


# ---------- Your manual Dropout ----------
def dropout_forward(x, dropout_prob=0.3, training=True):
    if not training:
        return x
    mask = (np.random.rand(*x.shape) > dropout_prob).astype(np.float32)
    out = (x.detach().cpu().numpy() * mask) / (1.0 - dropout_prob)
    return torch.from_numpy(out).to(x.device).float()

# ---------- PyTorch model using manual BN & Dropout ----------
class FashionMNISTManualBNDropout(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28*28, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 10)
        # gamma and beta for manual batchnorm
        self.gamma1 = nn.Parameter(torch.ones(512))
        self.beta1  = nn.Parameter(torch.zeros(512))
        self.gamma2 = nn.Parameter(torch.ones(512))
        self.beta2  = nn.Parameter(torch.zeros(512))
        self.dropout_prob = 0.3

    def forward(self, x, training=True):
        x = self.flatten(x)
        x = self.fc1(x)
        x = batchnorm_forward(x, self.gamma1, self.beta1)
        x = torch.relu(x)
        x = dropout_forward(x, dropout_prob=self.dropout_prob, training=training)

        x = self.fc2(x)
        x = batchnorm_forward(x, self.gamma2, self.beta2)
        x = torch.relu(x)
        x = dropout_forward(x, dropout_prob=self.dropout_prob, training=training)

        x = self.fc3(x)
        return x

# ---------- Training & Evaluation ----------
def train(dataloader, model, loss_fn, optimizer):
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        pred = model(X, training=True)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if batch % 100 == 0:
            print(f"Batch {batch}, Loss: {loss.item():.6f}")

def evaluate(dataloader, model, loss_fn):
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X, training=False)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= len(dataloader)
    accuracy = 100 * correct / len(dataloader.dataset)
    print(f"Evaluation Accuracy: {accuracy:.2f}%, Avg Loss: {test_loss:.6f}\n")

# ---------- Main ----------
def main():
    training_data = datasets.FashionMNIST(root="data", train=True, download=True, transform=ToTensor())
    test_data = datasets.FashionMNIST(root="data", train=False, download=True, transform=ToTensor())

    batch_size = 64
    train_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    model = FashionMNISTManualBNDropout().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    epochs = 5
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        train(train_loader, model, loss_fn, optimizer)
        evaluate(test_loader, model, loss_fn)

    torch.save(model.state_dict(), "fashionmnist_manualBN_dropout.pth")
    print("Saved model to fashionmnist_manualBN_dropout.pth")

if __name__ == "__main__":
    main()
