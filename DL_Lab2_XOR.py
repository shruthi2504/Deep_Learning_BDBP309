import numpy as np

from LAB2.DL_Lab2_forwardpass import ForwardPass



def main():

    # AND GATE
    print("AND GATE:")
    x1 = [[0, 0], [0, 1], [1, 0], [1, 1]]
    y1 = [np.array([0, 0, 0, 1])]
    layers = 2
    neurons = [2, 1]
    weights = [np.array([[1], [1]])]
    biases = [np.array([-1])]
    outputs1 = []
    for x in x1:
        outputx, _, _ = ForwardPass(x, layers, neurons, weights, biases,single=1).feed_forward()
        outputs1.append(outputx)
        print(f"Output for {x}: {outputx}")

    # OR GATE
    print("\nOR GATE:")
    x2 = [[0, 0], [0, 1], [1, 0], [1, 1]]
    y2 = [np.array([0, 1, 1, 1])]
    layers = 2
    neurons = [2, 1]
    weights = [np.array([[1], [1]])]
    biases = [np.array([0])]
    outputs2 = []
    for x in x2:
        outputx, _, _ = ForwardPass(x, layers, neurons, weights, biases,single=1).feed_forward()
        outputs2.append(outputx)
        print(f"Output for {x}: {outputx}")

    # XOR GATE
    print("\nXOR GATE:")
    X = [[0, 0], [0, 1], [1, 0], [1, 1]]
    y = np.array([[0, 1, 1, 0]])
    layers = 3
    neurons = [2, 2, 1]
    weights = [
        np.array([[1, 1],
                  [1, 1]]),
        np.array([[1],
                  [-2]])
    ]
    biases = [
        np.array([0, -1]),
        np.array([0])
    ]
    output = []
    for x in X:
        outputx, _, _ = ForwardPass(x, layers, neurons, weights, biases,single=1).feed_forward()
        print(f"Output for {x}: {outputx}")

if __name__ == "__main__":
    main()