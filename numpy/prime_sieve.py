import numpy as np


def prime(n=100):
    is_prime = np.ones((n,), dtype=bool)
    data = np.arange(n)

    is_prime[:2] = False
    n_max = (int)(np.sqrt(n))
    for j in range(2, n_max):
        is_prime[2*j::j] = False

    return data[is_prime]
