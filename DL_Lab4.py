# import numpy as np
#
#
# def softmax_func(z):
#     return np.exp(z)/np.sum(np.exp(z))
#
# def relu_func(z):
#     return np.maximum(0,z)
#
# def relu_derivative(z):
#     return np.where(z>0, 1, 0)
#
# def feed_forward(input, layers, neurons, weights):
#     input = np.array(input).reshape(1, -1)  # shape: (1, input_dim)
#
#     layer_inputs = []  # z values
#     layer_outputs = []  # a values
#
#     for l in range(layers - 1):
#         if weights[l] is None or weights[l].shape[0] != input.shape[1]:
#             print(f"The dimension of weights is not correct at layer {l + 1}")
#             return
#
#         z_l = np.dot(input, weights[l])
#         a_l = relu_func(z_l)
#
#         layer_inputs.append(z_l)
#         layer_outputs.append(a_l)
#
#         input = a_l  # Feed to next layer
#
#     # Final layer with softmax
#     z_last = np.dot(input, weights[layers - 1])
#     a_last = softmax_func(z_last)
#
#     layer_inputs.append(z_last)
#     layer_outputs.append(a_last)
#
#     return a_last, layer_inputs, layer_outputs
#
#
# def main():
#     X = np.array([
#         [0, 0, 1],
#         [1, 1, 1],
#         [1, 0, 1],
#         [0, 1, 1]
#     ])
#     y = np.array([0, 1, 1, 0])
#     layers =2
