import torch
from torch import nn
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

# Use GPU if available
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

# Training loop
def train(dataloader, model, loss_fn, optimizer):
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        pred = model(X)
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
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= len(dataloader)
    accuracy = 100 * correct / len(dataloader.dataset)
    print(f"Evaluation Accuracy: {accuracy:.2f}%, Avg Loss: {test_loss:.6f}\n")

# FashionMNIST model with BatchNorm and Dropout
class FashionMNISTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

def main():
    print("\n--- Running FashionMNIST with BatchNorm & Dropout ---")
    # Load data
    training_data = datasets.FashionMNIST(root="data", train=True, download=True, transform=ToTensor())
    test_data = datasets.FashionMNIST(root="data", train=False, download=True, transform=ToTensor())

    batch_size = 64
    train_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    # Initialize model, loss, optimizer
    model = FashionMNISTModel().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    epochs = 5
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        train(train_loader, model, loss_fn, optimizer)
        evaluate(test_loader, model, loss_fn)

    # Save model
    torch.save(model.state_dict(), "fashionmnist_model_bn_dropout.pth")
    print("Saved model to fashionmnist_model_bn_dropout.pth")

if __name__ == "__main__":
    main()
