#import statements
import numpy as np
import matplotlib.pyplot as plt
##Question1
#generating 100 equally spaced numbers from -10 to 10
z = np.linspace(-10, 10, 100)
#a)sigmoid function
#not used as it saturates at extreme ends so gradient doesn't update
def sigmoid_func(z):
    return 1 / (1 + np.exp(-z))
def sigmoid_derivative(z):
    return sigmoid_func(z) * (1 - sigmoid_func(z))

#b)tanh function
def tanh_func(z):
    return (np.exp(z)-np.exp(-z))/(np.exp(z)+np.exp(-z))
def tanh_derivative(z):
    return 1-(tanh_func(z)**2)

#c)ReLU(Reticular Linear Unit) function
def relu_func(z):
    return np.maximum(0,z)
def relu_derivative(z):
    return np.where(z>0, 1, 0)

#d)Leaky ReLU
def leaky_relu_func(z):
    return np.maximum(0.01*z,z)
def leaky_relu_derivative(z):
    return np.where(z>0, 1, 0.01)

#e)Softmax
def softmax_func(z):
    return np.exp(z)/np.sum(np.exp(z))
def softmax_derivative(z):
    s = softmax_func(z)
    jacobian_matrix = np.zeros((len(s),len(s)))
    for i in range(len(s)):
        for j in range(len(s)):
            if i == j:
                jacobian_matrix[i][j] = s[i]*(1-s[i])
            else:
                jacobian_matrix[i][j] = -s[i]*s[j]
    return jacobian_matrix
x5 = softmax_func(z)
y5 = softmax_derivative(z)
#checking the softmax probabilities
print("Sum of Softmax probabilities:",np.sum(x5))

#Question2
#a)
#Sigmoid function: min:,max:

def plot_observations(name,func,derv,z):
    x = func(z) #function output
    y = derv(z) #derivative ouput
    # if name != "softmax":
    plt.plot(z,x,color='red',label=f"{name} function")
    plt.plot(z,y,color='blue',label=f"{name} derivative")
    plt.title(f"{name} function and derivative")
    plt.xlabel("z")
    plt.ylabel("y(function and derivative)")
    plt.legend()
    plt.show()

    #Question2-Observations
    print("--------------------------------------------------------------")
    print(f"Observations of {name} plots and functions:")
    #a.max and min of each function
    print("Minimum value of the function:",np.min(x))
    print("Maximum value of the function:",np.max(x))
    #b.Is the output of the function zero-centred?
    mean = np.mean(x)
    if abs(mean) < 1e-3:
        print(f"The average output of the function {mean} thus it is zero centered") #Faster convergence
    else:
        print(f"The average output of the function is {mean} thus it is not zero centered")
    print("--------------------------------------------------------------")


plot_observations("sigmoid",sigmoid_func,sigmoid_derivative,z)
plot_observations("tanh",tanh_func,tanh_derivative,z)
plot_observations("relu",relu_func,relu_derivative,z)
plot_observations("leaky",leaky_relu_func,leaky_relu_derivative,z)
# plot_observations("softmax",softmax_func,softmax_derivative,z)

#c.What happens to the gradient when the input values are too small or too big

input_val_small = np.linspace(-100,-10,5)
input_val_medium = np.linspace(-10,10,5)
input_val_large = np.linspace(10,100,5)
input_val = np.concatenate((input_val_small,input_val_medium,input_val_large))

for val in input_val:
    print("-----------------------------------------------------------")
    print(f"Input value: {val}")
    print("The gradient for sigmoid function is",sigmoid_derivative(val))
    print("The gradient for tanh function is",tanh_derivative(val))
    print("The gradient for relu function is",relu_derivative(val))
    print("The gradient for leaky function is",leaky_relu_derivative(val))
    # print("The gradient for softmax function is",softmax_derivative(z))
##When input values are too big: Sigmoid and tanh(even though zero centered) are very small,ReLU and leaky ReLU is 1
#When input values are too small: Sigmoid and tanh(even though zero centered) are very small,ReLU is 0 and leaky ReLU is 0.01
#d.Relationship between sigmoid and tanh
z2 = np.linspace(-10, 10, 100)
plt.plot(z2, sigmoid_func(z2), label="Sigmoid")
# plt.plot(z2, sigmoid_derivative(z2), label="Sigmoid derivative")
plt.plot(z2, tanh_func(z2), label="Tanh")
# plt.plot(z2, tanh_derivative(z2), label="Tanh derivative")
plt.title("Sigmoid and Tanh with derivatives")
plt.xlabel("Input z2")
plt.ylabel("Output / Gradient")
plt.legend()
plt.show()
#Strong gradients around input 0 and saturates/flattens around extreme ends-no learning
#Gradient values range from 0-1 for sigmoid and -1 to 1 for tanh
















