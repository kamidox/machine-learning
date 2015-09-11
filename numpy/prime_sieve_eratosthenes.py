import numpy as np


def prime(n=100):
    is_prime = np.ones((n,), dtype=bool)
    data = np.arange(n)

    is_prime[:2] = False
    n_max = (int)(np.sqrt(n))
    for j in range(2, n_max):
        if is_prime[j]:
            is_prime[j**2::j] = False

    return data[is_prime]
