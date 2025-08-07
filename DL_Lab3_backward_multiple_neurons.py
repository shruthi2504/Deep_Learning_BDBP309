import numpy as np
np.random.seed(99)
#one hidden layer with 3 neurons
def relu_func(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return np.where(z > 0, 1, 0)

def forward(x,W1,b1,W2,b2):

    W2 = W2.reshape(-1,1)
    z1 = np.dot(x,W1) + b1 # for hidden layer with 3 neurons
    a1 = relu_func(z1)
    z2 = np.dot(a1,W2) + b2  #for output layer with 1 neuron
    f = z2

    forward_vals = {
        'x': x,
        'z1': z1, 'a1': a1,
        'z2': z2, 'f' : f
    }

    return f,forward_vals


def backward(f, f_true, forward_vals, W2, b2, W1, b1):
    x = forward_vals['x']
    z1 = forward_vals['z1']
    a1 = forward_vals['a1']
    z2 = forward_vals['z2']

    # derivatives
    # loss derivative
    # updating loss function
    # output layer
    dl_df = f - f_true
    df_dz2 = relu_derivative(z2)
    dz2_dW2 = a1.reshape(1, -1)
    dz2_da1 = W2.T
    dz2_db2 = 1

    # hidden layer
    da1_dz1 = relu_derivative(z1)
    dz1_dW1 = x.reshape(1, -1)
    dz1_da1 = 1

    # input layer
    dz1_dx = W1.T

    # global
    # output layer
    dl_dz2 = dl_df * df_dz2
    dl_dw2 = dl_dz2 * a1.reshape(-1, 1)  # shape becomes (3,1)

    # dl_dw2 = dl_dz2 * dz2_dW2
    dl_db2 = dl_dz2 * dz2_db2

    # hidden layer
    dl_da1 = (dl_dz2 * dz2_da1).flatten()
    dl_dz1 = dl_da1 * da1_dz1

    # FIXED: changed from element-wise multiplication to outer product for correct shape
    dl_dW1 = np.outer(x,dl_dz1)

    dl_db1 = dl_da1 * da1_dz1

    # input layer
    dl_dx = dl_dz1 @ dz1_dx  #matrix multiply instead of element-wise
    dl_dx1 = dl_dx[0]
    dl_dx2 = dl_dx[1]

    return {
        'dW2': dl_dw2,
        'db2': dl_db2,
        'dW1': dl_dW1,
        'db1': dl_db1
    }


print("Manual implementation of two layer(hidden+output) neural network:")
X = np.array([
    [1, 2],
    [0, 1],
    [2, 3]
])

# Corresponding targets
Y = np.array([1, 0.5, 2])
neurons = [3,1]
#he initialization
W1 = np.random.normal(0,np.sqrt(2/neurons[0]),size=(2,3))
W2 = np.random.normal(0, np.sqrt(2 / neurons[1]),size=(3,1))

b1 = np.random.uniform(0.1, 1.0, size=(3,))
b2 = np.array([0.1])

lr=0.01
epochs = 10
for epoch in range(epochs):
    total_loss = 0
    print('Epoch', epoch)

    for i in range(len(X)):
        x = X[i]
        f_true = Y[i]

        f, forward_vals = forward(x, W1, b1, W2, b2)
        loss = 0.5 * (f - f_true)**2
        total_loss += loss.item()

        grads = backward(f, f_true, forward_vals, W2, b2, W1, b1)

        # Update weights for this example (SGD)
        W2 -= lr * grads['dW2']
        b2 -= lr * grads['db2']
        W1 -= lr * grads['dW1']
        b1 -= lr * grads['db1']

        print(f"  Sample {i + 1}: Predicted = {f}, True = {f_true}, Loss = {loss}")

    avg_loss = total_loss / len(X)
    print(f"Average Loss: {avg_loss:}")
    print("----------------------------")



#checking with pytorch


print("Pytorch implementation of 2 layer neural network:")
#using pytorch
import torch
from torch import nn
model = nn.Sequential(
    nn.Linear(2, 3),  # Input to hidden
    nn.ReLU(),
    nn.Linear(3, 1)  # Hidden to output
)



# Manually set weights/biases to match NumPy
with torch.no_grad():
    model[0].weight.copy_(torch.tensor(W1.T))  # shape (3, 2)
    model[0].bias.copy_(torch.tensor(b1))
    model[2].weight.copy_(torch.tensor(W2.T))  # shape (1, 3)
    model[2].bias.copy_(torch.tensor(b2))


# Loss and optimizer
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

#  Training loop
epochs = 10
for epoch in range(epochs):
    total_loss = 0.0
    print(f"\nEpoch {epoch + 1}")

    for i in range(len(X)):
        x = torch.tensor(X[i], dtype=torch.float32)
        y_true = torch.tensor(Y[i], dtype=torch.float32).unsqueeze(0)
        optimizer.zero_grad()
        y_pred = model(x)
        loss = nn.MSELoss()(y_true, y_pred)


        loss.backward()
        optimizer.step()

        print(
            f"  Sample {i + 1}: Predicted = {y_pred[0]}, True = {y_true[0]}, Loss = {loss}")

        total_loss += loss.item()

    print(f"Average Loss: {total_loss / len(X)}")
    print("-----------------------------")




















