# -*- coding: utf-8 -*-
import scipy as sp
import matplotlib.pyplot as pl


def load_data(fname, delimiter=','):
    """ return the features x and result y as matrix """
    data = sp.loadtxt(fname, delimiter=delimiter)
    m, n = data.shape
    x = sp.asmatrix(data[:, range(0, n - 1)].reshape(m, n - 1))
    y = sp.asmatrix(data[:, n - 1].reshape(m, 1))
    return x, y


def plot_data(x, y):
    pl.clf()

    pl.scatter(x[:, 0][y == 1], x[:, 1][y == 1], s=50, c='r', marker='+')
    pl.scatter(x[:, 0][y == 0], x[:, 1][y == 0], s=50, c='b', marker='o')
    pl.xlabel('Exam 1 score')
    pl.ylabel('Exam 2 score')
    pl.legend(['Admitted', 'Not admitted'], loc='lower left')
    pl.show()


def sigmoid(z):
    """ Sigmoid Function """

    try:
        xdim, ydim = z.shape
    except ValueError:      # for array
        z = z.reshape(1, z.size)
    except AttributeError:  # for scalar value
        z = sp.array([z]).reshape(1, 1)
    finally:
        xdim, ydim = z.shape

    g = sp.zeros((z.shape)).view(sp.matrix)
    for i in range(xdim):
        for j in range(ydim):
            g[i, j] = (1 + sp.e ** (-z[i, j])) ** (-1)

    return g


def cost_function(theta, x, y):
    """ return cost and grad on each parameters

    cost = logistic regression cost function;
    grad = partial derivative of cost function on each theta;
    """

    z = sp.dot(x, theta)
    g = sigmoid(z)
    m, n = x.shape

    j = sp.sum((- sp.array(y)) * sp.log(g) - sp.array((1 - y)) * sp.log(1 - g)) / m

    # Compute partial derivatives of the Cost Function for each theta
    grad = sp.asmatrix(sp.zeros((n, 1)))
    for i in range(n):
        grad[i] = sp.sum(sp.array(g - y) * sp.array(x[:, i])) / m

    return j, grad

if __name__ == '__main__':
    print('============ Part 1: Plotting =============================')
    x, y = load_data('ex2/ex2data1.txt')
    plot_data(x, y)

    print('============ Part 2: Compute Cost and Gradient ============')
    m, n = x.shape
    x = sp.column_stack((sp.ones((m, 1)), x))
    init_theta = sp.asmatrix(sp.zeros((n + 1, 1)))
    cost, grad = cost_function(init_theta, x, y)
    print('Cost at initial theta: %s' % cost)
    print('Gradient at initial theta:\n %s' % grad)
