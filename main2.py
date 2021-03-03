import numpy as np
import matplotlib.pyplot as plt
from Perceptron import Perceptron

data = np.genfromtxt('sonar.all-data.txt', delimiter=',')
np.random.shuffle(data)

inputs = data[:, 0:60]
targets = data[:, 60]

n_inputs = 60
p1 = Perceptron(n_inputs)
alfa = 0.005

N = 200
E = np.zeros(N)
for i in range(N):
    p1.error_train_epoch(inputs, targets, alfa)
    E[i] = p1.error_value(inputs, targets)
    print(E[i])

plt.plot(E)
plt.ylabel('E')
plt.xlabel('epochy')
plt.show()
