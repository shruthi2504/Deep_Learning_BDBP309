import numpy as np
# #First case
# #initializing input values and weights
# x1 = np.array([0.2,-1.3,2,0.01])
# w1 = np.array([0.001,0.01,-0.005,-1.2])
# z1= np.dot(x1,w1)
#

class ForwardPass:
    def __init__(self, input, layers, neurons, weights, biases=None, single=None):
        self.input = input
        self.layers = layers
        self.neurons = neurons
        self.weights = weights
        self.biases = biases
        self.single = single

    @staticmethod
    def softmax_func(z):
        return np.exp(z) / np.sum(np.exp(z))

    @staticmethod
    def relu_func(z):
        return np.maximum(0, z)

    @staticmethod
    def relu_derivative(z):
        return np.where(z > 0, 1, 0)

    @staticmethod
    def activation_func(z):
        return (z > 0).astype(int)

    def feed_forward(self):
        self.input = np.array(self.input)
        if self.input.ndim == 1:
            self.input = self.input.reshape(1, -1)

        layer_inputs = []  # z values
        layer_outputs = []  # a values

        for l in range(self.layers - 2):
            if self.weights[l] is None or self.weights[l].shape[0] != self.input.shape[1]:
                print(f"The dimension of weights is not correct at layer {l + 1}")
                return

            if self.biases is not None:
                z_l = np.dot(self.input, self.weights[l]) + self.biases[l]
            else:
                z_l = np.dot(self.input, self.weights[l])

            if self.single is not None:
                a_l = self.activation_func(z_l)
            else:
                a_l = self.relu_func(z_l)

            layer_inputs.append(z_l)
            layer_outputs.append(a_l)
            self.input = a_l

        if self.biases is not None:
            z_last = np.dot(self.input, self.weights[self.layers - 2]) + self.biases[self.layers - 2]
        else:
            z_last = np.dot(self.input, self.weights[self.layers - 2])

        if self.single is not None:
            a_last = self.activation_func(z_last)
        else:
            a_last = self.relu_func(z_last)

        layer_inputs.append(z_last)
        layer_outputs.append(a_last)

        return a_last, layer_inputs, layer_outputs

    @staticmethod
    def user_input():
        input_str = input("Enter input vector (comma-separated): ")
        input_vector = [float(x.strip()) for x in input_str.split(',')]

        layers = int(input("Enter number of layers (including output layer): "))
        neurons_str = input(f"Enter number of neurons in each of the {layers} layers (comma-separated): ")
        neurons = np.array([int(x.strip()) for x in neurons_str.split(',')])

        weights = []
        input_dim = len(input_vector)
        for i in range(layers):
            rows = input_dim if i == 0 else neurons[i - 1]
            cols = neurons[i]
            print(f"Enter weights for Layer {i + 1} ({rows} x {cols} matrix):")
            w = []
            for r in range(rows):
                row_input = input(f"  Row {r + 1} (comma-separated): ")
                row = [float(x.strip()) for x in row_input.split(',')]
                if len(row) != cols:
                    raise ValueError(f"Expected {cols} values in row {r + 1}, got {len(row)}")
                w.append(row)
            weights.append(np.array(w))

        fp1 = ForwardPass(input_vector, layers, neurons, weights)
        output, layer_inputs, layer_outputs = fp1.feed_forward()

        print(f"The output of the forward pass is: {output}")

    @staticmethod
    def run(input, layers, neurons, weights, biases=None, single=None):
        fp = ForwardPass(input, layers, neurons, weights, biases, single)
        return fp.feed_forward()

def main():
    #example1
    input1 = [-2.4,1.2,-0.8,-1.1]
    layers1 = 4
    neurons1 = np.array([3,2,2,2])
    weights_11 = np.array([[0.1,0.1,0.1],
                          [0.1,0.1,0.1],
                          [0.1,0.1,0.1],
                          [0.1,0.1,0.1]])
    weights_12 = np.array([[0.001,0.001],
                          [0.001,0.001],
                          [0.001,0.001]])
    weights_13 = np.array([[0.01,0.01],[0.01,0.01]])
    weights_14 = np.array([[0.01,0.01],[0.01,0.01]])
    weights1 = [weights_11, weights_12, weights_13,weights_14]

    fp1 = ForwardPass(input1, layers1, neurons1, weights1)
    output1, layer_inputs1, layer_outputs1 = fp1.feed_forward()
    print(f"Example1: The output of the forward pass is: {output1}")


    #example2
    # x1 = np.array([0.2,1.3,2,0.01])
    input2 = np.array([55,10,25,3])
    layers2 = 6
    neurons2 = np.array([3,3,3,3,2,2])
    weights_21 = np.array([[0.1,0.1,0.1],
                          [0.1,0.1,0.1],
                          [0.1,0.1,0.1],
                          [0.1,0.1,0.1]])
    weights_22 = np.array([[0.001,0.001,0.001],
                          [0.001,0.001,0.001],
                          [0.001,0.001,0.001]])
    weights_23 = np.array([[0.01,0.001,0.001],
                          [0.001,0.1,0.01],
                          [0.01,0.001,0.0001]])
    weights_24 = np.array([[0.01,0.001,0.1],
                          [0.001,0.1,0.01],
                          [0.001,0.01,0.01]])
    weights_25 = np.array([[0.001,0.01],[0.01,0.001],[0.00001,0.001]])
    weights_26 = np.array([[0.01,0.01],[0.01,0.01]])
    weights2 = [weights_21, weights_22, weights_23,weights_24,weights_25,weights_26]

    fp2 = ForwardPass(input2, layers2, neurons2, weights2)
    output2, layer_inputs2, layer_outputs2 = fp2.feed_forward()
    print(f"Example1: The output of the forward pass is: {output2}")
    #example 3
    input3 = np.array([0.2,1.3,2])
    layers3 = 3
    neurons3 = np.array([2,2,2])
    weights_31 = np.array([[0.1,0.1,],
                           [-2,0.43],
                           [0.22,0.1]])
    weights_32 = np.array([[-0.1,0.1],
                           [0.56,1.2]])
    weights_33 = np.array([[0.001,4.5],
                           [-2.2,1.8]])
    weights3 = [weights_31, weights_32, weights_33]

    fp3 = ForwardPass(input3, layers3, neurons3, weights3)
    output3, layer_inputs3, layer_outputs3 = fp3.feed_forward()
    print(f"Example1: The output of the forward pass is: {output3}")

    ForwardPass.user_input()


if __name__ == "__main__":
    main()





#as a function
# import numpy as np
#
# def softmax_func(z):
#     return np.exp(z) / np.sum(np.exp(z))
#
# def relu_func(z):
#     return np.maximum(0, z)
#
# def feed_forward(input, layers, weights):
#     input = np.array(input)
#     if input.ndim == 1:
#         input = input.reshape(1, -1)  # shape becomes (1, features)
#
#     layer_inputs = []  # z values
#     layer_outputs = []  # a values
#
#     for l in range(layers - 2):
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
#     z_last = np.dot(input, weights[layers - 2])
#     a_last = softmax_func(z_last)
#
#     layer_inputs.append(z_last)
#     layer_outputs.append(a_last)
#
#     return a_last, layer_inputs, layer_outputs

