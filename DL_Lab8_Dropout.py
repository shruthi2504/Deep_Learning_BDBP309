import numpy as np

def dropout_forward(x, dropout_prob=0.5, training=True):
    """
    Forward pass for dropout.
    x: input activations
    dropout_prob: fraction of units to drop
    training: if False, no dropout is applied (test mode)
    """
    if not training:
        return x, None
    mask = (np.random.rand(*x.shape) > dropout_prob).astype(np.float32)
    out = (x * mask) / (1.0 - dropout_prob) #1-dropout_prob is for scaling
    return out, mask

def dropout_backward(dout, mask, dropout_prob=0.5, training=True):
    """
    Backward pass for dropout.
    dout: upstream gradients
    mask: dropout mask from forward pass
    """
    if not training:
        return dout
    dx = (dout * mask) / (1.0 - dropout_prob)
    return dx

#Testing with example

# input activation of a layer with 5 neurons
x = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
print("Original x:", x)

# Forward pass with dropout
dropout_prob = 0.4
out, mask = dropout_forward(x, dropout_prob=dropout_prob, training=True)
print("\nDropout mask:", mask)
print("Output after dropout (forward):", out)

# gradient from next layer is all ones (for simplicity)
dout = np.ones_like(x)

# Backward pass
dx = dropout_backward(dout, mask, dropout_prob=dropout_prob, training=True)
print("\nGradient after dropout (backward):", dx)

# Test mode (no dropout)
out_test, _ = dropout_forward(x, dropout_prob=dropout_prob, training=False)
print("\nOutput in test mode (no dropout):", out_test)
