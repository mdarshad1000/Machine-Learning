import numpy as np


def gradient_descent(x, y):
    iterations = 1000
    learning_rate = 0.001
    m = b = 0
    n = len(x)

    for i in range(iterations):
        y_predicted = m * x + b
        cost = (1/n) * sum([val ** 2 for val in (y - y_predicted)])
        m_derivative = -(2/n) * sum(x(y - y_predicted))
        b_derivative = -(2/n) * sum(y - y_predicted)
        m = m - learning_rate * m_derivative
        b = b - learning_rate * b_derivative


x = np.array([1, 2, 3, 4, 5])
y = np.array([4, 6, 8, 10, 12])
