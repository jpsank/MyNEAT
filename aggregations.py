"""
Has the built-in aggregation functions.
Largely copied from NEAT-Python.
"""

from operator import mul
from functools import reduce

from myneat.math_util import mean, median2


def product_aggregation(x):  # note: 'x' is a list or other iterable
    return reduce(mul, x, 1.0)


def sum_aggregation(x):
    return sum(x)


def max_aggregation(x):
    return max(x)


def min_aggregation(x):
    return min(x)


def maxabs_aggregation(x):
    return max(x, key=abs)


def median_aggregation(x):
    return median2(x)


def mean_aggregation(x):
    return mean(x)


aggregation_functions = {
    "product": product_aggregation,
    "sum": sum_aggregation,
    "max": max_aggregation,
    "min": min_aggregation,
    "maxabs": maxabs_aggregation,
    "median": median_aggregation,
    "mean": mean_aggregation
}

