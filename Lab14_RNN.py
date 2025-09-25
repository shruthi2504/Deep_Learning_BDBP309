import numpy as np

def tanh_func(z):
    return (np.exp(z)-np.exp(-z))/(np.exp(z)+np.exp(-z))

def h_t_cal(w_hh,w_xh,x_t,h_pt):
    h_t = (w_hh@h_pt) + (w_xh@x_t)
    h_t = tanh_func(h_t)
    return h_t
def y_t_cal(w_hy,h_t):
    y_t = w_hy@h_t
    return y_t

def input():
    h0 = np.array([
        [0],
        [0],
        [0]
    ])

    w_xh = np.array([
        [0.5,-0.3],
        [0.8,0.2],
        [0.1,0.4]
    ])

    w_hy = np.array([
        [1,-1,0.5],
        [0.5,0.5,-0.5]
    ])

    w_hh = np.array([
        [-.1,0.4,0],
        [-0.2,0.3,0.2],
        [0.05,-0.1,0.2]
    ])

    x_1 = np.array([
        [1],
        [2]
    ])

    x_2 = np.array([
        [-1],
        [1]
    ])

    sample1 = [x_1,x_2]

    samples = [sample1]

    return h0,w_xh,w_hy,w_hh,sample1,samples


def main():

    h0,w_xh,w_hy,w_hh,sample1,samples = input()

    h_prev = h0

    for s in range(len(samples)):
        for t in range(len(sample1)):
            h_t = h_t_cal(w_hh,w_xh,sample1[t],h_prev)
            y_t = y_t_cal(w_hy, h_t)

            print(f"Step {t+1}:")
            print(f"h_t:\n{h_t}")
            print(f"y_t:\n{y_t}")
            print("-------------------------------")

            h_prev = h_t


if __name__ == '__main__':
    main()
