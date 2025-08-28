import numpy as np

def conv2d(X,filters,pad,stride):
    n_h, n_w, n_c = X.shape
    num_filters, f_h, f_w, f_c = filters.shape
    out_h = (n_h + 2*pad - f_h)//stride + 1
    out_w = (n_w + 2*pad - f_w)//stride + 1
    out = np.zeros((out_h, out_w, num_filters))
    X_pad = np.pad(X, ((pad,pad),(pad,pad),(0,0)), constant_values=0)

    for k in range(num_filters):
        f = filters[k]
        for i in range(out_h):
            for j in range(out_w):
                X_slice = X_pad[i*stride:i*stride+f_h, j*stride:j*stride+f_w, :]
                out[i,j,k] = np.sum(X_slice * f)
    return out

def relu(X):
    return np.maximum(X, 0)

def max_pooling(X,size,stride):
    n_h, n_w, n_c = X.shape
    out_h = (n_h - size) // stride + 1
    out_w = (n_w - size) // stride + 1
    out = np.zeros((out_h, out_w, n_c))

    for c in range(n_c):
        for i in range(out_h):
            for j in range(out_w):
                X_slice = X[i*stride:i*stride+size, j*stride:j*stride+size, c]
                out[i, j, c] = np.max(X_slice)
    return out

def flatten(X):
    return X.flatten().reshape(1,-1)

def fc(X, W, b):
    return np.dot(X, W) + b

def softmax_func(z):
    exp_shift = np.exp(z - np.max(z))
    return exp_shift / np.sum(exp_shift)

def cnn_forward(X, filters, fc_weights, fc_bias,padding,stride,size):

    print("Original Input Shape: ", X.shape)

    conv_output = conv2d(X, filters, padding, stride)
    print("Convolution Output Shape: ", conv_output.shape)

    conv_output2 = relu(conv_output)
    print('Output shape After Activation Shape:' , conv_output2.shape)

    conv_pooling = max_pooling(conv_output,size,stride)
    print('Output shape After Max Pooling: ',conv_pooling.shape)

    flatten_output = flatten(conv_pooling)
    print('Output shape After Flattening: ',flatten_output.shape)

    fc_output = fc(flatten_output, fc_weights, fc_bias)
    final_prob = softmax_func(fc_output)
    print("Final Probability: ", final_prob)


def main():
    # Input: 4x4 image with 3 channels
    X = np.array([
        [[1,0,1], [2,1,0], [3,0,1], [4,1,0]],
        [[5,1,0], [6,0,1], [7,1,0], [8,0,1]],
        [[9,0,1], [10,1,0], [11,0,1], [12,1,0]],
        [[13,1,0], [14,0,1], [15,1,0], [16,0,1]]
    ])

    # Filters: 2 filters, each 2x2 with 3 channels
    filters = np.array([
        [[[1,0,1],[0,1,0]], [[-1,0,1],[1,1,0]]],
        [[[0,1,0],[1,0,1]], [[1,0,-1],[0,1,0]]]
    ])

    # Fully connected weights and bias
    fc_weights = np.random.randn(8, 3) 
    fc_bias = np.random.randn(1, 3)
    padding = 0
    stride = 1
    size = 2

    cnn_forward(X, filters, fc_weights, fc_bias,padding,stride,size)


if __name__ == '__main__':
    main()



































