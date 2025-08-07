# #using sympy
# import sympy as sym
# def gradient(expression,input_values):
#     expr = sym.sympify(expression)
#     vars = expr.free_symbols
#     gradients ={}
#     for var in vars:
#         grad = sym.diff(expr,var)
#         d_var_d_expr = grad.subs(input_values).evalf()
#         gradients[var] = d_var_d_expr
#         if d_var_d_expr == 0:
#             print(f"∂f/∂{var} : 0.0")
#         else:
#             print(f"∂f/∂{var} : {d_var_d_expr}")
#     print(gradients)
# w0, x0, w1, x1, w2 = sym.symbols("w0 x0 w1 x1 w2")
# net_expr = w0 * x0 + w1 * x1 + w2
# expr  =  1 / (1 + sym.exp(-net_expr))
# input_values = {w0: 2, x0: -1, w1: -3, x1: -2, w2: -3}
# gradient(expr, input_values)
# print("-------------------------------------")
# #example 2
# x,y,z,w = sym.symbols("x y z w")
# f = ((x * y) + sym.Max(z, w)) * 2
# input_values2 = {x:3,y:-4,z:2,w:-1}
# gradient(f, input_values2)
####FOR A SINGLE NEURON
#hardcoding and implementing from scratch
import numpy as np

def relu_func(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return np.where(z > 0, 1, 0)

def one_neuron_backward(x1, w1, x2, w2,b,f_true):
    # Forward
    f1 = x1 * w1
    f2 = x2 * w2
    f3 = f1 + f2
    f4 = f3 + b
    f = relu_func(f4)

    # Backward
    df_df4 = relu_derivative(f4)

    #local * global gradients- chain rule
    df_dx1 = df_df4 * w1
    df_dw1 = df_df4 * x1
    df_dx2 = df_df4 * w2
    df_dw2 = df_df4 * x2
    df_db = df_df4 * 1

    #loss function
    loss = ((f-f_true)**2)/2

    return {
        "df/dx1": df_dx1,
        "df/dw1": df_dw1,
        "df/dx2": df_dx2,
        "df/dw2": df_dw2,
        "df/db": df_db,
        "f" : f,
        "loss" : loss
    }
# Test
print("Manual implementation of one neuron network:")
x1, w1 = 2, 3
x2, w2 = 4, 5
b= -1
f_true = 27
epochs = 25
learning_rate = 0.01
for epoch in range(epochs):
    grads = one_neuron_backward(x1, w1, x2, w2, b,f_true)
    f = grads["f"]
    loss = grads["loss"]

    #updating loss function
    dl_df = f - f_true
    #updating weights
    w1 -= learning_rate * dl_df * grads["df/dw1"]
    w2 -= learning_rate * dl_df * grads["df/dw2"]
    b -= learning_rate * dl_df * grads["df/db"]

    print(f"Loss for epoch {epoch} : {loss}\n w1: {w1}\n w2: {w2}\n b: {b}\n f_predicted: {f}")

###testing with pytorch to compare with manual
import torch
from torch import nn

print("Pytorch implementation of one neuron network:")

x = torch.tensor([2.0, 4.0])
y_true = torch.tensor([27.0])
# Model: f = relu(w1*x1 + w2*x2 + b)
model = nn.Sequential(
    nn.Linear(2, 1),  # w1*x1 + w2*x2 + b
    nn.ReLU()
)
model[0].weight.data = torch.tensor([[3.0, 5.0]])
model[0].bias.data = torch.tensor([-1.0])
loss_fn = nn.MSELoss()
# Training loop
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
for epoch in range(25):
    optimizer.zero_grad()

    y_pred = model(x)  # Forward pass
    loss = loss_fn(y_pred, y_true)  # Compute loss

    loss.backward()  #  performing Backward pass
    optimizer.step()  # Updating weights

    print(f"Epoch {epoch}")
    print(f"  Loss: {loss.item()}")
    print(f"  Prediction: {y_pred.item()}")
    print(f"  Weights: {model[0].weight.data}")
    print(f"  Bias: {model[0].bias.data}")
    print("------------------")
