# -*- coding: utf-8 -*-
import scipy as sp
import matplotlib.pyplot as pl


def plot_data(x, y):
    pl.clf()
    pl.xlabel('Populations of city in 10,000s')
    pl.ylabel('Profit in $10,0000s')
    pl.scatter(x, y, c='r', marker='x')


def compute_cost(x, y, theta):
    m = len(y)
    delta = sp.dot(x, theta) - y
    return sp.sum(delta ** 2) / (2 * m)


def gradient_descent(x, y, theta, alpha, iterations):
    m = len(y)
    cost_history = sp.zeros((iterations, 1))
    n = len(theta)
    new_theta = sp.zeros((2, 1))
    for loop in range(iterations):
        new_theta = theta
        delta = sp.dot(x, theta) - y
        for j in range(n):
            new_theta[j] = theta[j] - (alpha * (sp.sum(delta * (x[:, j].reshape(m, 1))) / m))
        theta = new_theta

        cost_history[loop] = compute_cost(x, y, theta)

    return theta, cost_history


def main():
    # =============== Step 1: Plotting data =================
    print('Plotting data ...')
    data = sp.loadtxt('../ex1/ex1data1.txt', delimiter=',')
    m = sp.shape(data)[0]
    x = data[:, 0].reshape(m, 1)
    y = data[:, 1].reshape(m, 1)
    plot_data(x, y)
    pl.show()

    # =============== Step 2: Graident Descent =================
    x = sp.column_stack((sp.ones((m, 1)), x))
    theta = sp.zeros((2, 1))
    iterations = 1500
    alpha = 0.01
    print('Initial cost %f' % compute_cost(x, y, theta))
    theta, j_history = gradient_descent(x, y, theta, alpha, iterations)
    print('Theta found by gradient descent: %f %f' % (theta[0, 0], theta[1, 0]))

    # Plot the linear fit
    plot_data(x[:, 1], y)
    pl.plot(x[:, 1], sp.dot(x, theta), linewidth=2)
    pl.show()


if __name__ == '__main__':
    main()
