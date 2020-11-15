"""
Contains configuration for genomes, species, populations, etc., and generators for float, bool, and string values.
Also contains counters for id's and innovation numbers.
Configuration parameters themselves are largely copied from NEAT-Python.
"""

from dataclasses import dataclass
import random


class Counter:
    def __init__(self):
        self.last = 0

    def __call__(self):
        self.last += 1
        return self.last


def clamp(value, low, high):
    return max(min(value, high), low)


@dataclass
class FloatConfig:
    init_mean: float
    init_stdev: float
    max_value: float
    min_value: float
    mutate_power: float
    mutate_rate: float
    replace_rate: float
    init_type: str = "gauss"

    def new(self):
        if self.init_type == "gauss":
            return clamp(random.gauss(self.init_mean, self.init_stdev), self.min_value, self.max_value)
        if self.init_type == "uniform":
            min_value = max(self.min_value, (self.init_mean - (2 * self.init_stdev)))
            max_value = min(self.max_value, (self.init_mean + (2 * self.init_stdev)))
            return random.uniform(min_value, max_value)

    def mutate(self, value):
        # mutate_rate is usually no lower than replace_rate, and frequently higher -
        # so put first for efficiency
        r = random.random()
        if r < self.mutate_rate:
            return clamp(value + random.gauss(0.0, self.mutate_power), self.min_value, self.max_value)

        if r < self.replace_rate + self.mutate_rate:
            return self.new()

        return value


@dataclass
class BoolConfig:
    mutate_rate: float
    default: bool = None
    rate_to_true_add: float = 0.0
    rate_to_false_add: float = 0.0

    def new(self):
        return bool(random.random() < 0.5) if self.default is None else self.default

    def mutate(self, value):
        if self.mutate_rate > 0:
            # The mutation operation *may* change the value but is not guaranteed to do so
            if random.random() < self.mutate_rate:
                return random.random() < 0.5
        return value


@dataclass
class StringConfig:
    options: [str]
    mutate_rate: float
    default: str = None

    def new(self):
        if self.default is not None:
            return self.default
        return random.choice(self.options)

    def mutate(self, value):
        if self.mutate_rate > 0:
            if random.random() < self.mutate_rate:
                return random.choice(self.options)
        return value


@dataclass
class Config:
    """
    Contains configuration and counters for a simulation
    """

    pop_size = 100

    # node activation options
    activation_config = StringConfig(
        default="tanh",
        mutate_rate=0.05,
        options=["sigmoid", "tanh", "sin", "gauss", "identity"]
    )

    # node aggregation options
    aggregation_config = StringConfig(
        default="sum",
        mutate_rate=0.05,
        options=["sum", "product", "max", "min"]
    )

    # node bias options
    bias_config = FloatConfig(
        init_mean=0.0,
        init_stdev=1.0,
        max_value=30.0,
        min_value=-30.0,
        mutate_power=0.5,
        mutate_rate=0.7,
        replace_rate=0.1
    )

    # connection add/remove rates
    conn_add_prob = 0.5
    conn_delete_prob = 0.5

    # connection enable options
    enabled_config = BoolConfig(
        default=True,
        mutate_rate=0.01
    )

    # feed_forward = False
    initial_connection = "full"

    # node add/remove rates
    node_add_prob = 0.2
    node_delete_prob = 0.2

    # network parameters
    # num_hidden = 0
    num_inputs = 1
    num_outputs = 1

    # node response options
    response_config = FloatConfig(
        init_mean=1.0,
        init_stdev=0.0,
        max_value=30.0,
        min_value=-30.0,
        mutate_power=0.0,
        mutate_rate=0.0,
        replace_rate=0.0
    )

    # connection weight options
    weight_config = FloatConfig(
        init_mean=0.0,
        init_stdev=1.0,
        max_value=30.0,
        min_value=-30.0,
        mutate_power=0.5,
        mutate_rate=0.8,
        replace_rate=0.1
    )

    # genome compatibility options
    compatibility_excess_coefficient = 1.0  # c1
    compatibility_disjoint_coefficient = 1.0  # c2
    compatibility_weight_coefficient = 0.5  # c3
    compatibility_threshold = 8.0

    def __init__(self):
        self.innovation_counter = Counter()
        self.node_id_counter = Counter()
        self.genome_id_counter = Counter()
        self.species_id_counter = Counter()


default_config = Config()

