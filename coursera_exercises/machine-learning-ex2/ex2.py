# -*- coding: utf-8 -*-
import scipy as sp
import matplotlib.pyplot as pl
import scipy.optimize as op


def load_data(fname, delimiter=','):
    """ return the features x and result y as matrix """
    data = sp.loadtxt(fname, delimiter=delimiter)
    m, n = data.shape
    x = sp.asmatrix(data[:, range(0, n - 1)].reshape(m, n - 1))
    y = sp.asmatrix(data[:, n - 1].reshape(m, 1))
    return x, y


def plot_data(x, y):
    pl.clf()

    # BUG on MAC: There's bugs in pylab of Mac Version
    # pos_x0 and pos_x1 must be ndarray but now matrix, or it will crash in Mac
    # Althought the matrix is running well on Windows
    n = x.shape[1]
    pos_x0 = sp.array(x[:, n - 2][y == 1])
    pos_x1 = sp.array(x[:, n - 1][y == 1])
    neg_x0 = sp.array(x[:, n - 2][y == 0])
    neg_x1 = sp.array(x[:, n - 1][y == 0])
    pl.scatter(pos_x0, pos_x1, s=50, c='r', marker='+')
    pl.scatter(neg_x0, neg_x1, s=50, c='b', marker='o')
    pl.xlabel('Exam 1 score')
    pl.ylabel('Exam 2 score')
    pl.legend(['Admitted', 'Not admitted'], loc='lower left')


def plot_decision_boundary(theta, x, y):
    plot_data(x, y)

    if x.shape[1] <= 3:
        # Only need 2 points to define a line, so choose two endpoints
        plot_x = [sp.amin(x[:, 1]) - 2, sp.amax(x[:, 1]) + 2]

        # Calculate the decision boundary line
        # REMARK: prove the the plot_y calculators
        # let sigmoid(z) = 0.5 be the decision boundary
        # => e^{-z} = 1
        # => z = 0
        # => X * theta = 0
        # => x_0 * theta_0 + x_1 * theta_1 + x_2 * theta_2 = 0
        # => x_2 = - theta_2 * (x_1 * theta_1 + x_0 * theta_0)
        # => x_2 = - theta_2 * (x_1 * theta_1 + theta_0)
        plot_y = (-1 / sp.array(theta[2])) * (sp.array(theta[1]) * plot_x + theta[0])
        # Plot, and adjust axes for better viewing
        pl.plot(plot_x, plot_y)
        # Legend, specific for the exercise
        pl.legend(['Admitted', 'Not admitted', 'Decision Boundary'], loc='lower left')


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

    g = sp.clip(g, a_min=0.0000000001, a_max=0.999999999)
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
    grad = sp.zeros((n, 1)).flatten()
    for i in range(n):
        grad[i] = sp.sum(sp.array(g - y) * sp.array(x[:, i])) / m

    return j, grad


def cost_function_without_grad(theta, x, y):
    z = sp.dot(x, theta)
    g = sigmoid(z)
    m, n = x.shape

    j = sp.sum((- sp.array(y)) * sp.log(g) - sp.array((1 - y)) * sp.log(1 - g)) / m

    return j


def err_handler(t, flag):
    print('Floating point error (%s), with flag %s' % (t, flag))


def main():
    saved_handler = sp.seterrcall(err_handler)
    saved_err = sp.seterr(all='call')

    print('============ Part 1: Plotting =============================')
    x, y = load_data('ex2/ex2data1.txt')
    plot_data(x, y)
    pl.show()

    print('============ Part 2: Compute Cost and Gradient ============')
    m, n = x.shape
    x = sp.column_stack((sp.ones((m, 1)), x))
    init_theta = sp.asmatrix(sp.zeros((n + 1, 1)))
    cost, grad = cost_function(init_theta, x, y)
    print('Cost at initial theta: %s' % cost)
    print('Gradient at initial theta:\n %s' % grad)

    print('============ Part 3: Optimizing minimize ====================')
    # res = op.minimize(cost_function, init_theta, args=(x, y), jac=True, method='Newton-CG')
    res = op.minimize(cost_function_without_grad, init_theta, args=(x, y), method='Powell')
    # print('Cost at theta found by fmin: %s' % cost)
    print('Result by minimize:\n%s' % res)
    plot_decision_boundary(res.x, x, y)
    pl.show()

    print('============ Part 4: Optimizing fmin ====================')
    res = op.fmin(cost_function_without_grad, init_theta, args=(x, y))
    # print('Cost at theta found by fmin: %s' % cost)
    print('Result by fmin:\n%s' % res)
    plot_decision_boundary(res, x, y)
    pl.show()

    sp.seterrcall(saved_handler)
    sp.seterr(**saved_err)

if __name__ == '__main__':
    main()
