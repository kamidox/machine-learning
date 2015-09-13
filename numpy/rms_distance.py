# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt


def rms_distance(n_stories=1000, t_max=200):
    """ Worked Example: diffusion using a random walk algorithm

    Let us consider a simple 1D random walk process: at each time step a walker
    jumps right or left with equal probability.
    We are interested in finding the typical distance from the origin of a
    random walker after t left or right jumps? We are going to simulate many
    “walkers” to find this law, and we are going to do so using array computing
    tricks: we are going to create a 2D array with the “stories” (each walker
    has a story) in one direction, and the time in the other

    ----------- t_max -----------------
    |1 |1 |-1|1 |-1|1
    |----------------------------------
    |-1|1 |1 |-1|-1|1
    |
    n_stories
    |
    |
    |
    ------------------------------------
    """

    t = np.arange(t_max)
    steps = 2 * np.random.random_integers(0, 1, (n_stories, t_max)) - 1
    positions = np.cumsum(steps, axis=1)
    distance = positions ** 2
    mean_distance = np.mean(distance, axis=0)

    return (t, mean_distance)


def rms_plot(fname='../rms_distance.png'):
    """plot the rms distance"""

    t, mean_distance = rms_distance()
    plt.clf()
    plt.title('Root Mean Square Distance')
    plt.plot(t, np.sqrt(mean_distance), 'g.', t, np.sqrt(t), 'r-')
    plt.xlabel(r'$t$')
    plt.ylabel(r'$\sqrt{\left[ (\delta x)^2 \right]}$')
    plt.savefig(fname)

if __name__ == '__main__':
    rms_plot()
