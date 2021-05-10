"""
Has the built-in activation functions.
Largely copied from NEAT-Python.
"""
import math


def sigmoid_activation(x):
    x = max(-60.0, min(60.0, 5.0 * x))
    return 1.0 / (1.0 + math.exp(-x))


def tanh_activation(x):
    x = max(-60.0, min(60.0, 2.5 * x))
    return math.tanh(x)


def sin_activation(x):
    x = max(-60.0, min(60.0, 5.0 * x))
    return math.sin(x)


def gauss_activation(x):
    x = max(-3.4, min(3.4, x))
    return math.exp(-5.0 * x ** 2)


def relu_activation(x):
    return x if x > 0.0 else 0.0


def softplus_activation(x):
    x = max(-60.0, min(60.0, 5.0 * x))
    return 0.2 * math.log(1 + math.exp(x))


def identity_activation(x):
    return x


def clamped_activation(x):
    return max(-1.0, min(1.0, x))


def inv_activation(x):
    try:
        x = 1.0 / x
    except ArithmeticError: # handle overflows
        return 0.0
    else:
        return x


def log_activation(x):
    x = max(1e-7, x)
    return math.log(x)


def exp_activation(x):
    z = max(-60.0, min(60.0, x))
    return math.exp(x)


def abs_activation(x):
    return abs(x)


def hat_activation(x):
    return max(0.0, 1 - abs(x))


def square_activation(x):
    return x ** 2


def cube_activation(x):
    return x ** 3


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
