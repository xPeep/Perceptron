import numpy as np
from Perceptron import Perceptron

n_inputs = 2
x = np.array([[-1, -1], [1, -1], [-1, 1], [1, 1]])
t = np.array([-1, 1, 1, 1])
alfa = 0.1

p1 = Perceptron(n_inputs)

N = 10
for i in range(N):
    p1.error_train_epoch(x, t, alfa)
    E = p1.error_value(x, t)
    print(E)
