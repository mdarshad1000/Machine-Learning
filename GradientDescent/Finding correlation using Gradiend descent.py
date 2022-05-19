import numpy as np
import pandas as pd
from sklearn import linear_model
import math


def prediction_using_sklearn():
    df = pd.read_csv('test_scores.csv')
    reg = linear_model.LinearRegression()
    reg.fit(df[['math']], df['cs'])

    return reg.coef_, reg.intercept_


def prediction_using_gradient_descent(x, y):
    m_curr = 0
    b_curr = 0
    iterations = 1000000
    cost_previous = 0
    n = len(x)
    learning_rate = 0.0002

    for item in range(iterations):
        y_predicted = m_curr * x + b_curr
        cost = (1/n) * sum([value ** 2 for value in (y - y_predicted)])
        md = -(2/n) * sum(x*(y - y_predicted))
        bd = -(2/n) * sum(y - y_predicted)
        m_curr = m_curr - learning_rate * md
        b_curr = b_curr - learning_rate * bd
        if math.isclose(cost, cost_previous, rel_tol=1e-20):
            break
        cost_previous = cost
        print(f"m {m_curr}, intercept {b_curr}, cost = {cost}, iteration {item}")

    return m_curr, b_curr


if __name__ == '__main__':
    df = pd.read_csv('test_scores.csv')
    x = np.array(df['math'])
    y = np.array(df['cs'])

    m, b = prediction_using_gradient_descent(x, y)
    print(f"Using Gradient Descent coefficient is {m} and intercept is {b}")

    m_sklearn, b_sklearn = prediction_using_sklearn()
    print(f"Using sklearn coefficient is {m_sklearn} and intercept is {b_sklearn}")
