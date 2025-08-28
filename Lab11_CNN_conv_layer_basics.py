import numpy as np



def conv_op():
    #input
    X = np.array([
        [[3], [0], [1], [2], [7], [4]],
        [[1], [5], [8], [9], [3], [1]],
        [[2], [7], [2], [5], [1], [3]],
        [[0], [1], [3], [1], [7], [8]],
        [[4], [2], [1], [6], [2], [8]],
        [[2], [4], [5], [2], [3], [9]]
    ])
    #filter
    f = np.array([
        [[1], [0], [-1]],
        [[1], [0], [-1]],
        [[1], [0], [-1]]
    ])

    #intialize stride and pad
    stride = 1
    pad = 0
    #Dimensions of input and filter
    n_h, n_w, n_c = X.shape
    f_h, f_w, f_c = f.shape

    #Output dimensions
    out_h = (n_h + 2 * pad - f_h) // stride + 1
    out_w = (n_w + 2 * pad - f_w) // stride + 1

    #Initialzing output matrix
    out = np.zeros((out_h, out_w))

    #Convolution operation
    for i in range(out_h):
        for j in range(out_w):
            X_slice = X[i:i+f_h,j:j+f_w,:] #i:,j: are row and column selections of X for a slice-dim match the filter dim
            #Element wise multiplication
            out[i,j] = np.sum(X_slice*f)


    print(out)

conv_op()

def convo_op_pad():
    #input
    X = np.array([
        [[2], [3], [7], [4], [6]],
        [[6], [6], [9], [8], [7]],
        [[3], [4], [8], [3], [8]],
        [[7], [8], [3], [6], [6]],
        [[4], [2], [1], [8], [3]],
    ])

    f = np.array([
        [[1], [-1]],
        [[0], [-1]],
    ])
    #intialize stride and pad
    stride = 2
    pad = 1
    #Dimensions of input and filter
    n_h, n_w, n_c = X.shape
    f_h, f_w, f_c = f.shape

    #Output dimensions
    out_h = (n_h + 2 * pad - f_h) // stride + 1
    out_w = (n_w + 2 * pad - f_w) // stride + 1

    #Initialzing output matrix
    out = np.zeros((out_h, out_w))

    #adding padding
    X_pad = np.pad(X, ((pad, pad), (pad, pad), (0, 0)), constant_values=0)

    #Convolution operation
    for i in range(out_h):
        for j in range(out_w):
            X_slice = X_pad[i*stride:i*stride+f_h,j*stride:j*stride+f_w,:] #i:,j: are row and column selections of X for a slice-dim match the filter dim
            #Element wise multiplication
            out[i,j] = np.sum(X_slice*f)


    print(out)

convo_op_pad()


def conv3d_op():
    # Input: 4x4 image with 3 channels
    X = np.array([
        [[1, 2, 1], [2, 0, 1], [3, 1, 0], [4, 2, 1]],
        [[5, 1, 0], [6, 2, 1], [7, 0, 2], [8, 1, 1]],
        [[9, 0, 1], [10, 1, 0], [11, 2, 1], [12, 1, 0]],
        [[13, 1, 0], [14, 0, 1], [15, 1, 1], [16, 0, 2]]
    ])
    print("X3d shape:",X.shape)

    # Filter: 2x2 filter with 3 channels (must match input channels)
    f = np.array([
        [[1, 0, 1], [0, 1, 0]],
        [[-1, 0, 1], [1, 1, 0]]
    ])

    stride = 1
    pad = 0
    n_h, n_w, n_c = X.shape
    f_h, f_w, f_c = f.shape

    # Output dimensions
    out_h = (n_h + 2 * pad - f_h) // stride + 1
    out_w = (n_w + 2 * pad - f_w) // stride + 1

    # Initialize output
    out = np.zeros((out_h, out_w))

    # Pad input if needed
    X_pad = np.pad(X, ((pad, pad), (pad, pad), (0, 0)), constant_values=0)

    # Convolution
    for i in range(out_h):
        for j in range(out_w):
            X_slice = X_pad[i * stride:i * stride + f_h, j * stride:j * stride + f_w, :]
            out[i, j] = np.sum(X_slice * f)

    print("3D Convolution output:\n", out)

conv3d_op()

def generalised_convo_op(X,f,pad,stride):
    n_h, n_w, n_c = X.shape
    f_h, f_w, f_c = f.shape

    # Output dimensions
    out_h = (n_h + 2 * pad - f_h) // stride + 1
    out_w = (n_w + 2 * pad - f_w) // stride + 1

    # Initialize output
    out = np.zeros((out_h, out_w))

    # Pad input if needed
    X_pad = np.pad(X, ((pad, pad), (pad, pad), (0, 0)), constant_values=0)

    # Convolution
    for i in range(out_h):
        for j in range(out_w):
            X_slice = X_pad[i * stride:i * stride + f_h, j * stride:j * stride + f_w, :]
            out[i, j] = np.sum(X_slice * f)

    print("3D Convolution output:\n", out)

def testing_convo_op():
    print("FINAL:")
    X = np.array([
        [[1, 2, 1], [2, 0, 1], [3, 1, 0], [4, 2, 1]],
        [[5, 1, 0], [6, 2, 1], [7, 0, 2], [8, 1, 1]],
        [[9, 0, 1], [10, 1, 0], [11, 2, 1], [12, 1, 0]],
        [[13, 1, 0], [14, 0, 1], [15, 1, 1], [16, 0, 2]]
    ])

    # Filter: 2x2 filter with 3 channels (must match input channels)
    f = np.array([
        [[1, 0, 1], [0, 1, 0]],
        [[-1, 0, 1], [1, 1, 0]]
    ])

    stride = 1
    pad = 0
    generalised_convo_op(X,f,pad,stride)

    #example two
    print("FINAL 2:")

    X = np.array([
        [[1, 2, 1], [2, 0, 1], [3, 1, 0], [4, 2, 1]],
        [[5, 1, 0], [6, 2, 1], [7, 0, 2], [8, 1, 1]],
        [[9, 0, 1], [10, 1, 0], [11, 2, 1], [12, 1, 0]],
        [[13, 1, 0], [14, 0, 1], [15, 1, 1], [16, 0, 2]]
    ])


    # Filter: 2x2 filter with 3 channels (must match input channels)
    f = np.array([
        [[1, 0, 1], [0, 1, 0]],
        [[-1, 0, 1], [1, 1, 0]]
    ])

    stride = 2
    pad = 1
    generalised_convo_op(X,f,pad,stride)

testing_convo_op()

def final_convo_op(X,filters,pad,stride):
    print("FINAL with multiple filters:")
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

def main():
    # Input: 4x4 image with 3 channels
    X = np.array([
        [[1, 0, 1], [2, 1, 0], [3, 0, 1], [4, 1, 0]],
        [[5, 1, 0], [6, 0, 1], [7, 1, 0], [8, 0, 1]],
        [[9, 0, 1], [10, 1, 0], [11, 0, 1], [12, 1, 0]],
        [[13, 1, 0], [14, 0, 1], [15, 1, 0], [16, 0, 1]]
    ])

    # Filters: 2 filters, each 2x2 with 3 channels
    filters = np.array([
        [  # Filter 1
            [[1, 0, 1], [0, 1, 0]],
            [[-1, 0, 1], [1, 1, 0]]
        ],
        [  # Filter 2
            [[0, 1, 0], [1, 0, 1]],
            [[1, 0, -1], [0, 1, 0]]
        ]
    ])

    pad = 0
    stride = 1
    result = final_convo_op(X, filters, pad, stride)
    print("Output:\n", result)


if __name__ == "__main__":
    main()