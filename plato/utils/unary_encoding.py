"""Implements unary encoding, used by Google's RAPPOR, as the local differential privacy mechanism.

References:

Wang, et al. "Optimizing Locally Differentially Private Protocols," ATC USENIX 2017.

Erlingsson, et al. "RAPPOR: Randomized Aggregatable Privacy-Preserving Ordinal Response,"
ACM CCS 2014.

"""

import numpy as np


def encode(x: np.ndarray):
    # 1bit
    # x[x > 0] = 1
    # x[x <= 0] = 0
    # return x、

    # 2bit
    x[(x > 0.675)] = 1.15
    x[(x > 0) & (x <= 0.675)] = 0.315
    x[(x > -0.675) & (x <= 0)] = -0.315
    x[x <= -0.675] = -1.15
    return x


def randomize(bit_array: np.ndarray, epsilon):
    """
    The default unary encoding method is symmetric.
    """
    assert isinstance(bit_array, np.ndarray)
    return symmetric_unary_encoding(bit_array, epsilon)


def symmetric_unary_encoding(bit_array: np.ndarray, epsilon):
    # p = np.e**(epsilon / 2) / (np.e**(epsilon / 2) + 1)
    # q = 1 / (np.e**(epsilon / 2) + 1)
    # return produce_random_response(bit_array, p, q)

    # 2bit
    p = np.e**(epsilon / 2) / (np.e**(epsilon / 2) + 3)
    q = round((1-p)/3, 4)
    p = 1 - q * 3
    return k_random_response(bit_array, [-1.15, -0.315, 0.315, 1.15], p, q)

def k_random_response(value: np.ndarray, values, p, q):
    """
    the k-random response
    :param value: current value
    :param values: the possible value
    :param p: prob
    :return:
    """
    if not isinstance(values, list):
        raise Exception("The values should be list")
    k = len(values)
    replace_arrays = []
    out_array = np.zeros(value.shape)
    for i in range(k):
        probs = [q] * k
        probs[i] = p
        replace_arrays.append(np.random.choice(values, size=value.shape, replace=True, p=probs))
    for j in range(k):
        out_array[value == values[j]] = replace_arrays[j][value == values[j]]

    return out_array


def optimized_unary_encoding(bit_array: np.ndarray, epsilon):
    p = 1 / 2
    q = 1 / (np.e**epsilon + 1)
    return produce_random_response(bit_array, p, q)


def produce_random_response(bit_array: np.ndarray, p, q=None):
    """Implements random response as the perturbation method."""
    q = 1 - p if q is None else q

    p_binomial = np.random.binomial(1, p, bit_array.shape)
    q_binomial = np.random.binomial(1, q, bit_array.shape)
    return np.where(bit_array == 1, p_binomial, q_binomial)
