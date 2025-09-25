import string

import numpy as np



#
# def string_to_one_hot(inputs: np.ndarray) -> np.ndarray:
#     char_to_index = {char: i for i, char in enumerate(string.ascii_uppercase)}
#
#     one_hot_inputs = []
#     for row in inputs:
#         one_hot_list = []
#         for char in row:
#             if char.upper() in char_to_index:
#                 one_hot_vector = np.zeros((len(string.ascii_uppercase), 1))
#                 one_hot_vector[char_to_index[char.upper()]] = 1
#                 one_hot_list.append(one_hot_vector)
#         one_hot_inputs.append(one_hot_list)
#
#     return np.array(one_hot_inputs)
#


def string_to_one_hot(sequence,alphabet="ABCDEFGHIGKLMNOPQRTSUVWXYZ"):

    alpha_size = len(alphabet)
    char_to_index = {char:i for i, char in enumerate(alphabet)}

    one_hot_seq = []
    for char in sequence:
        vec = np.zeros((alpha_size,1))
        vec[char_to_index[char]] = 1
        one_hot_seq.append(vec)
    return one_hot_seq

class InputLayer:
    def __init__(self, inputs : np.ndarray,hidden_size: int):
        self.inputs = inputs
        self.weights = np.random.uniform(low=0,high=1,size=(hidden_size,len(inputs[0])))

    def get_input(self,t: int):
        return self.inputs[t]

    def weighted_sum(self,t):
        return self.weights @ self.inputs[t]

class RNN:
    def __init__(self,input_size:int,hidden_size:int,output_size:int):
        #weights
        self.weights_hx = np.random.uniform(low=0,high=1,size=(hidden_size,input_size))
        self.weights_hh = np.random.uniform(low=0,high=1,size=(hidden_size,hidden_size))
        self.weights_hy = np.random.uniform(low=0,high=1,size=(output_size,hidden_size))

        #biases
        self.bias_h = np.zeros((hidden_size,1))
        self.bias_y = np.zeros((output_size,1))

        #hidden states-h0
        self.h = np.zeros((hidden_size,1))


    def step(self,x_t):
        #hidden state calculation and update
        self.h = np.tanh(self.weights_hx @ x_t + self.weights_hh @ self.h + self.bias_h)

        #ouput prediction-y
        z = self.weights_hy @ self.h + self.bias_y
        exp_scores = np.exp(z-np.max(z))
        y_t = exp_scores / np.sum(exp_scores)
        return y_t

    def forward(self,inputs):
        outputs = []
        for x_t in inputs:
            y_t = self.step(x_t)
            outputs.append(y_t)
        return outputs


def main():
    #example data
    sample_inputs = "ABC"
    one_hot_seq = string_to_one_hot(sample_inputs)
    alpha_size = len(string.ascii_uppercase)
    hidden_size = 10

    rnn = RNN(input_size=alpha_size,hidden_size=hidden_size,output_size=alpha_size)

    outputs = rnn.forward(one_hot_seq)

    for t,pred in enumerate(outputs):
        predicted_char = string.ascii_uppercase[np.argmax(pred)]

        print(f"Step {t}: predicted {predicted_char}")


if __name__ == "__main__":
    main()



