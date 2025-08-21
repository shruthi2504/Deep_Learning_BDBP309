import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split

def load_data(file="combined.npy"):
    data = np.load("combined.npy")

    print("Original shape:", data.shape)

    split_num = 943

    X = data[:split_num].T
    print("Input data(landmark) shape:", X.shape)


    X_tensor = torch.tensor(X, dtype=torch.float32)

    Y = data[split_num:,:].T
    print("Output data(target) shape:", Y.shape)

    Y_tensor = torch.tensor(Y, dtype=torch.float32)

    dataset = TensorDataset(X_tensor, Y_tensor)

    num_samples = len(dataset)

    return X,Y,dataset
def dataloaders(dataset):
    num_samples = len(dataset)
    train_size = int(0.8 * num_samples)
    val_size = int(0.1 * num_samples)
    test_size = num_samples - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])


    batch_size = 32

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, val_loader, test_loader, train_size, val_size, test_size

def ffn_model(input_dim,output_dim,hidden_dim,dropout_prob):

    model = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.BatchNorm1d(hidden_dim), # Tried with BatchNorm after ReLU but not much difference is observed
        nn.ReLU(),
        # nn.BatchNorm1d(hidden_dim),
        nn.Dropout(dropout_prob),
        nn.Linear(hidden_dim, hidden_dim),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU(),
        # nn.BatchNorm1d(hidden_dim),
        nn.Dropout(dropout_prob),
        nn.Linear(hidden_dim, output_dim)
    )

    for layer in model:
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
    return model


def train_model(model, train_loader, val_loader, test_loader, train_size, val_size, test_size,lr):
    criterion = nn.MSELoss()                # regression
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    num_epochs = 1000

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for X_batch, Y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, Y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * X_batch.size(0)

        epoch_loss = running_loss / train_size
    #     print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {epoch_loss:.4f}")
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {epoch_loss:.4f}")

    return model,criterion,optimizer


def model_run(lr, dropout_prob):
    X, Y, dataset = load_data()
    train_loader, val_loader, test_loader, train_size, val_size, test_size = dataloaders(dataset)
    input_dim = X.shape[1]
    output_dim = Y.shape[1]
    hidden_dim = 512

    model = ffn_model(input_dim, output_dim, hidden_dim, dropout_prob)

    model, criterion, optimizer = train_model(model, train_loader, val_loader, test_loader, train_size, val_size,
                                              test_size, lr)

    # Evaluate only on validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for X_batch, Y_batch in val_loader:
            outputs = model(X_batch)
            loss = criterion(outputs, Y_batch)
            val_loss += loss.item() * X_batch.size(0)
    val_loss /= val_size

    return model, criterion, val_loss, test_loader, test_size


def main():
    learning_rates = [0.01, 0.005, 0.001]
    dropout_probs = [0.3, 0.5]

    results = []

    for lr in learning_rates:
        for dropout_prob in dropout_probs:
            print("=" * 50)
            print(f"Tuning with lr={lr}, dropout={dropout_prob}")
            model, criterion, val_loss, test_loader, test_size = model_run(lr, dropout_prob)
            results.append({"lr": lr, "dropout": dropout_prob, "val_loss": val_loss})

    results_df = pd.DataFrame(results)
    print("\nHyperparameter tuning results:")
    print(results_df)

    # pick best hyperparameter
    best = results_df.loc[results_df["val_loss"].idxmin()]
    print(f"\nBest hyperparameters: lr={best['lr']}, dropout={best['dropout']}")

    # final test eval with best hyperparameters
    print("\nFinal evaluation on test set:")
    model, criterion, val_loss, test_loader, test_size = model_run(best['lr'], best['dropout'])

    test_loss = 0.0
    with torch.no_grad():
        for X_batch, Y_batch in test_loader:
            outputs = model(X_batch)
            loss = criterion(outputs, Y_batch)
            test_loss += loss.item() * X_batch.size(0)
    test_loss /= test_size
    print(f"Test Loss: {test_loss:.4f}")
    #
    # model, criterion, val_loss, test_loader, test_size = model_run(0.01, 0.31)
    # test_loss = 0.0
    # with torch.no_grad():
    #     for X_batch, Y_batch in test_loader:
    #         outputs = model(X_batch)
    #         loss = criterion(outputs, Y_batch)
    #         test_loss += loss.item() * X_batch.size(0)
    # test_loss /= test_size
    # print(f"Test Loss: {test_loss:.4f}")
    #


if __name__ == "__main__":
    main()












