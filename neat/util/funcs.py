"""
Has the built-in activation and aggregation functions.
Also commonly used functions not available in the Python3 standard library.
"""
from statistics import mean, median, stdev, variance
from operator import mul
from functools import reduce
import math
import numpy as np


def clip(x: float, l: float, u: float) -> float:
    return l if x < l else u if x > u else x


# ACTIVATIONS

def sigmoid_activation(x: float) -> float:
    x = clip(5 * x, -60, 60)
    return 1.0 / (1.0 + math.exp(-x))


def tanh_activation(x: float) -> float:
    x = clip(2.5 * x, -60, 60)
    return math.tanh(x)


def sin_activation(x: float) -> float:
    x = clip(5 * x, -60, 60)
    return math.sin(x)


def gauss_activation(x: float) -> float:
    x = clip(x, -3.4, 3.4)
    return math.exp(-5.0 * x ** 2)


def relu_activation(x: float) -> float:
    return x if x > 0.0 else 0.0


def softplus_activation(x: float) -> float:
    x = clip(5 * x, -60, 60)
    return 0.2 * math.log(1 + math.exp(x))


def identity_activation(x: float) -> float:
    return x


def clamped_activation(x: float) -> float:
    return clip(x, -1.0, 1.0)


def inv_activation(x: float) -> float:
    try:
        x = 1.0 / x
    except ArithmeticError: # handle overflows
        return 0.0
    else:
        return x


def log_activation(x: float) -> float:
    x = max(1e-7, x)
    return math.log(x)


def exp_activation(x: float) -> float:
    x = clip(x, -60, 60)
    return math.exp(x)


def abs_activation(x: float) -> float:
    return abs(x)


def hat_activation(x: float) -> float:
    return max(0.0, 1 - abs(x))


def square_activation(x: float) -> float:
    return x ** 2


def cube_activation(x: float) -> float:
    return x ** 3


# AGGREGATIONS

def softmax_aggregation(values) -> float:
    e = np.exp(values)
    return e / e.sum()


def product_aggregation(values) -> float:
    return reduce(mul, values, 1.0)


def sum_aggregation(values) -> float:
    return sum(values)


def max_aggregation(values) -> float:
    return max(values)


def min_aggregation(values) -> float:
    return min(values)


def maxabs_aggregation(values) -> float:
    return max(values, key=abs)


def median_aggregation(values) -> float:
    return median(values)


def mean_aggregation(values) -> float:
    return mean(values)


# Activations take a single value and return a single value.
activation_defs = {
    "sigmoid": sigmoid_activation,
    "tanh": tanh_activation,
    "sin": sin_activation,
    "gauss": gauss_activation,
    "relu": relu_activation,
    "softplus": softplus_activation,
    "identity": identity_activation,
    "clamped": clamped_activation,
    "inv": inv_activation,
    "log": log_activation,
    "exp": exp_activation,
    "abs": abs_activation,
    "hat": hat_activation,
    "square": square_activation,
    "cube": cube_activation,
}


# Aggregations take a list of values and return a single value.
aggregation_defs = {
    "product": product_aggregation,
    "sum": sum_aggregation,
    "max": max_aggregation,
    "min": min_aggregation,
    "maxabs": maxabs_aggregation,
    "median": median_aggregation,
    "mean": mean_aggregation
}


# Lookup table for commonly used {value} -> value functions.
stat_functions = {
    'min': min,
    'max': max,
    'mean': mean,
    'median': median
}
