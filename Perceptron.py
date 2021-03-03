import numpy as np


class Perceptron:
    def __init__(self, n_inputs):
        #self.w = np.zeros((n_inputs + 1, 1))
        self.w = np.random.randn(n_inputs + 1, 1)

    def agregate(self, x):
        output = self.w[0]
        for i in range(len(x)):
            output = output + self.w[i + 1] * x[i]

        return output

    def activate(self, y_a):
        if y_a >= 0:
            return 1
        else:
            return -1

    def calc_output(self, x):
        return self.activate(self.agregate(x))

    def hebb_iter(self, x, t):
        x = np.append(1, x)
        for i in range(len(self.w)):
            self.w[i] = self.w[i] + x[i] * t

    def hebb_epoch(self, inputs, targets):
        n = len(targets)
        for i in range(n):
            self.hebb_iter(inputs[i, :], targets[i])

    def error_train_iter(self, x, t, alfa):
        y = self.calc_output(x)
        e = t - y
        x = np.append(1, x)
        for i in range(len(self.w)):
            self.w[i] = self.w[i] + alfa * x[i] * e

    def error_train_epoch(self, inputs, targets, alfa):
        n = len(targets)
        for i in range(n):
            self.error_train_iter(inputs[i, :], targets[i], alfa)

    def error_value(self, inputs, targets):
        n = len(targets)
        E = 0
        for i in range(n):
            y = self.calc_output(inputs[i, :])
            e = targets[i] - y
            E = E + e * e

        E = E / n
        return E
