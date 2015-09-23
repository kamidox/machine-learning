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
    # pl.show()

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
    pl.legend(["Linear Regression", "Training Data"], loc='lower right')
    # pl.show()

    # Predict values for population sizes of 35,000 and 70,000
    predict1 = sp.dot(sp.array([1, 3.5]), theta)
    print('For population = 35,000, we predict a profit of %f' % (predict1 * 10000))
    predict2 = sp.dot(sp.array([1, 7]), theta)
    print('For population = 70,000, we predict a profit of %f' % (predict2 * 10000))

    # ============= Part 4: Visualizing J(theta_0, theta_1) =============
    print('Visualizing J(theta_0, theta_1) ...')

    # Grid over which we will calculate J
    theta0_vals = sp.linspace(-10, 10, 100)
    theta1_vals = sp.linspace(-1, 4, 100)

    # initialize j_vals to a matrix of 0's
    j_vals = sp.zeros((len(theta0_vals), len(theta1_vals)))

    # Fill out j_vals
    for i in range(len(theta0_vals)):
        for j in range(len(theta1_vals)):
            t = (sp.array([theta0_vals[i], theta1_vals[j]])).reshape(2, 1)
            j_vals[i, j] = compute_cost(x, y, t)

    # Because of the way meshgrids work in the surf command, we need to
    # transpose j_vals before calling surf, or else the axes will be flipped
    j_vals = j_vals.T
    # Contour plot
    # Plot j_vals as 15 contours spaced logarithmically between 0.01 and 100
    pl.clf()
    c = pl.contour(theta0_vals, theta1_vals, j_vals, sp.logspace(-2, 3, 30))
    pl.clabel(c, fontsize=9)
    pl.xlabel(r'$\theta_0$')
    pl.ylabel(r'$\theta_1$')
    pl.scatter(theta[0], theta[1], s=50, c='r', marker='x')
    pl.show()

if __name__ == '__main__':
    main()
