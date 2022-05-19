import numpy as np


def gradient_descent(x,y):
    m_now = b_now = 0
    iterations = 1000
    n = len(x)
    learning_rate = 0.08

    for i in range(iterations):
        y_predicted = m_now * x + b_now
        cost = (1 / (2*n)) * sum([value ** 2 for value in (y - y_predicted)])
        m_derivative = -(2/n) * sum(x * (y - y_predicted))
        b_derivative = -(2/n) * sum(y - y_predicted)
        m_now = m_now - learning_rate * m_derivative
        b_now = b_now - learning_rate * b_derivative
        print(f"m {m_now}, b {b_now}, iteration {i}, cost {cost}")


x = np.array([1, 2, 3, 4, 5])
y = np.array([5, 7, 9, 11, 13])

gradient_descent(x, y)