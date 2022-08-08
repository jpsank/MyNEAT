"""
Defines blueprints for creating, modifying, and applying functions to float, bool, and string values.
"""

from dataclasses import dataclass
import random
from typing import *

from neat.model import *
from neat.util.funcs import clip


class Blueprint:
    """ Abstract base class for all blueprint types. """
    
    def create(self): raise NotImplementedError
    def mutate(self, value): return value
    def copy(self, value): return value
    def crossover(self, a, b): return a if random.random() < 0.5 else b
    def distance(self, a, b): return abs(a - b)


# --------------- PRIMITIVE CONTROLLERS ---------------

@dataclass
class FloatBP(Blueprint):
    """ Blueprint for float values. """

    init_mean: float
    init_stdev: float
    max_value: float
    min_value: float
    mutate_power: float
    mutate_rate: float
    replace_rate: float
    init_type: str = "gauss"

    def create(self) -> float:
        if self.init_type == "gauss":
            return clip(random.gauss(self.init_mean, self.init_stdev), self.min_value, self.max_value)
        if self.init_type == "uniform":
            min_value = max(self.min_value, (self.init_mean - (2 * self.init_stdev)))
            max_value = min(self.max_value, (self.init_mean + (2 * self.init_stdev)))
            return random.uniform(min_value, max_value)
    
    def mutate(self, value) -> float:
        # mutate_rate is usually no lower than replace_rate, and frequently higher, so put first for efficiency
        r = random.random()
        if r < self.mutate_rate:
            return clip(value + random.gauss(0.0, self.mutate_power), self.min_value, self.max_value)

        if r < self.replace_rate + self.mutate_rate:
            return self.create()

        return value


@dataclass
class BoolBP(Blueprint):
    """ Blueprint for boolean values. """

    mutate_rate: float
    default: bool = None
    # rate_to_true_add: float = 0.0
    # rate_to_false_add: float = 0.0

    def create(self) -> bool:
        return bool(random.random() < 0.5) if self.default is None else self.default

    def mutate(self, value) -> bool:
        # The mutation operation *may* change the value but is not guaranteed to do so
        if self.mutate_rate > 0 and random.random() < self.mutate_rate:
            return random.random() < 0.5
        return value
    

@dataclass
class StringBP(Blueprint):
    """ Blueprint for string values. """
    
    options: 'list[str]'
    mutate_rate: float
    default: str = None

    def create(self) -> str:
        return random.choice(self.options) if self.default is None else self.default

    def mutate(self, value) -> str:
        if self.mutate_rate > 0 and random.random() < self.mutate_rate:
            return random.choice(self.options)
        return value
    
    def distance(self, a: str, b: str) -> float:
        return 0 if a == b else 1
