import numpy as np

def batch_stats(x):
    """
    Compute mean and variance for each feature across the batch.
    x: shape (batch_size, num_features)
    """
    mean = np.mean(x,axis=0)
    var = np.var(x,axis=0)
    return mean, var

def normalize(x,mean,var, eps=1e-5):
    """
    Normalize the batch using mean & variance.
    """
    std = np.sqrt(var + eps)  # use sqrt(var), add eps to avoid divide-by-zero
    return (x - mean) / std


def affine_transform(x_hat,gamma,beta):
    """
    Scale (gamma) and shift (beta) after normalization.
    """
    return (x_hat*gamma) + beta

def layer_normalization(x,gamma,beta, eps=1e-5):
    mean = np.mean(x, axis=1, keepdims=True)
    var = np.var(x, axis=1, keepdims=True)

    # Normalize
    x_hat = (x - mean) / np.sqrt(var+eps)

    # Scale + shift (gamma/beta are per-feature params)
    out = gamma * x_hat + beta
    return out,mean,var


def update_running_stats(running_mean, running_var, batch_mean, batch_var, momentum=0.9):
    """
    Update running averages of mean and variance across batches.
    (Keeps a moving average across batches.
    Used later in inference (test mode), when you don’t have the batch)
    """
    new_running_mean = momentum * running_mean + (1 - momentum) * batch_mean
    new_running_var = momentum * running_var + (1 - momentum) * batch_var
    return new_running_mean, new_running_var


def batchnorm():

    print("============BATCH NORMALIZATION==========")

    # Simulate one forward pass
    np.random.seed(0)
    x = np.random.randn(5, 3)   # batch_size=5, num_features=3

    gamma = np.ones(3)          # learnable scale
    beta = np.zeros(3)          # learnable shift

    # Initialize running stats
    running_mean = np.zeros(3)
    running_var = np.ones(3)

    # Simple SGD learning rate
    lr = 0.01

    # Fake "training loop"
    for step in range(5):
        # Input batch
        x = np.random.randn(5, 3)

        # 1. Batch stats
        batch_mean, batch_var = batch_stats(x)

        # 2. Update running stats
        running_mean, running_var = update_running_stats(running_mean, running_var,
                                                         batch_mean, batch_var)

        # 3. Normalize
        x_hat = normalize(x, batch_mean, batch_var)

        # 4. Affine transform
        out = affine_transform(x_hat, gamma, beta)

        # ------------------------
        # Fake loss: mean squared output (just for demo)
        # ------------------------
        loss = np.mean(out**2)

        # Gradients wrt gamma, beta
        dgamma = np.sum(out * x_hat, axis=0) / x.shape[0]
        dbeta = np.sum(out, axis=0) / x.shape[0]

        # Update params (SGD)
        gamma -= lr * dgamma
        beta -= lr * dbeta

        print(f"Step {step+1}")
        print("Batch mean:", batch_mean)
        print("Batch var:", batch_var)
        print("Output mean:", out.mean(axis=0))
        print("Output var:", out.var(axis=0))
        print("Running mean:", running_mean)
        print("Running var:", running_var)
        print("gamma:", gamma)
        print("beta:", beta)
        print("loss:", loss, "\n")

def layernorm():
#layer normalization
    print("============LAYER NORMALIZATION==========")

    np.random.seed(1)
    x = np.random.randn(5, 3)   # batch_size=5, num_features=3

    gamma = np.ones((1,3))      # shape matches features
    beta = np.zeros((1,3))
    for step in range(5):
        # Input batch
        x = np.random.randn(5, 3)
        # Simple SGD learning rate
        lr = 0.01

        # LayerNorm forward
        out, ln_mean, ln_var = layer_normalization(x, gamma, beta)

        # Fake loss (mean squared output)
        loss = np.mean(out**2)

        # Gradients wrt gamma, beta (per feature)
        dgamma = np.sum(out * (x - ln_mean) / np.sqrt(ln_var + 1e-5), axis=0) / x.shape[0]
        dbeta = np.sum(out, axis=0) / x.shape[0]

        # Update params
        gamma -= lr * dgamma
        beta -= lr * dbeta

        print(f"Step {step+1}")
        print("Input mean per sample:", ln_mean.flatten())
        print("Input var per sample:", ln_var.flatten())
        print("Output mean:", out.mean(axis=1))   # should be ~0 per sample
        print("Output var:", out.var(axis=1))     # should be ~1 per sample
        print("gamma:", gamma)
        print("beta:", beta)
        print("loss:", loss, "\n")



def main():
    batchnorm()
    layernorm()

if __name__ == "__main__":
    main()


#
# # 1. Batch stats
# batch_mean, batch_var = batch_stats(x)
#
# # 2. Update running stats
# running_mean, running_var = update_running_stats(running_mean, running_var,
#                                                  batch_mean, batch_var)
#
# # 3. Normalize
# x_hat = normalize(x, batch_mean, batch_var)
#
# # 4. Affine transform
# out = affine_transform(x_hat, gamma, beta)
#
# print("Input:\n", x)
# print("\nBatch mean:", batch_mean)
# print("Batch var:", batch_var)
# print("\nNormalized output:\n", out)
#
# print("Mean after BatchNorm:", out.mean(axis=0))
# print("Var after BatchNorm:", out.var(axis=0))
