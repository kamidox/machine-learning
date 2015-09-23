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
    new_theta = sp.zeros((n, 1))
    for loop in range(iterations):
        new_theta = theta
        delta = sp.dot(x, theta) - y
        for j in range(n):
            new_theta[j] = theta[j] - (alpha * (sp.sum(delta * (x[:, j].reshape(m, 1))) / m))
        theta = new_theta

        cost_history[loop] = compute_cost(x, y, theta)

    return theta, cost_history


def feature_normalize(x):
    """ normalized_value = (x_i - mean(x)) / (max(x) - min(x)) or
        normalized_value = (x_i - mean(x)) / numpy.std(x)
    """
    n = x.shape[1]
    mu = sp.zeros((n, 1))
    sigma = sp.zeros((n, 1))

    for i in range(n):
        mu[i] = sp.mean(x[:, i])
        sigma[i] = sp.std(x[:, i])
        x[:, i] = (x[:, i] - mu[i]) / sigma[i]

    return x, mu, sigma


def normal_equation(x, y):
    """ normal equation: theta = (X.T * X)^(-1) * X.T * y """

    theta = sp.dot(sp.dot(sp.mat(sp.dot(x.T, x)).I, x.T), y)
    return theta.reshape(x.shape[1], 1)


def main():
    # =============== Part 1: Gradient Descent with one variable =============
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

    # ============= Step 3: Visualizing J(theta_0, theta_1) =============
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
    # pl.show()

    # ============= Part 2: Gradient Descent with multi variables =============
    # ============= Step 1: Feature Normalization =============
    print('Part 2: Gradient Descent with multi variables')
    data = sp.loadtxt('../ex1/ex1data2.txt', delimiter=',')
    print('Step1: Feature Normalization ...')
    x = data[:, range(2)]
    y = data[:, 2]
    m = len(y)
    (x, mu, sigma) = feature_normalize(x)

    # ============= Step 2: Gradient Descent on multi variables =============
    print('Step2: Running gradient descent for multi variables')
    # add intercept term to x
    x = sp.column_stack((sp.ones((m, 1)), x))
    alpha = 0.0001
    iterations = 400
    theta = sp.zeros((3, 1))
    (theta, j_history) = gradient_descent(x, y, theta, alpha, iterations)
    # print result
    print('Theta compute from gradient descent: \n%s' % theta)

    # Predicted price of a 1650 sq-ft, 3 br house
    hourse = sp.array([1, (1650 - mu[0]) / sigma[0], (3 - mu[1]) / sigma[1]])
    price = sp.dot(hourse, theta)
    print('Predicted price of a 1650 sq-ft, 3 br house (using gradient descent): %s' % price)

    # ============= Step 3: Plot the convergence graph =============
    alpha1 = 0.0003
    theta1 = sp.zeros((3, 1))
    (theta1, j_history1) = gradient_descent(x, y, theta1, alpha1, iterations)
    alpha2 = 0.001
    theta2 = sp.zeros((3, 1))
    (theta2, j_history2) = gradient_descent(x, y, theta2, alpha2, iterations)
    print('Step 3: Plot the convergence graph')
    pl.clf()
    pl.plot(range(len(j_history)), j_history, c='r')
    pl.plot(range(len(j_history1)), j_history1, c='g')
    pl.plot(range(len(j_history2)), j_history2, c='b')
    pl.legend(['alpha=0.0001', 'alpha=0.0003', 'alpha=0.001'])
    pl.xlabel('Number of iterations')
    pl.ylabel('Cost')
    pl.show()

    # ============= Step 4: Using Normal Equation =============
    print('Step 4: Using Normal Equation')
    data = sp.loadtxt('../ex1/ex1data2.txt', delimiter=',')
    print('Step1: Feature Normalization ...')
    x = data[:, range(2)]
    y = data[:, 2]
    m = len(y)
    x = sp.column_stack((sp.ones((m, 1)), x))

    # Calculate the parameters from the normal equation
    theta = normal_equation(x, y)
    print('Theta computed from the normal equations: \n%s' % theta)

    # Estimate the price of a 1650 sq-ft, 3 br house
    hourse = sp.array([1, 1650, 3])
    price = hourse * theta
    print('Predicted price of a 1650 sq-ft, 3 br house (using normal equations):\n%s' % price)

if __name__ == '__main__':
    main()
