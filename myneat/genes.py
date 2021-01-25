import random
from dataclasses import dataclass

from myneat.myneat.config import Config


@dataclass
class NodeGene:
    """
    Base class for CPPN node genes
    out = activation(bias + response * aggregation(inputs))
    """
    key: int  # node primary key
    response: float
    bias: float
    activation: str
    aggregation: str

    @staticmethod
    def new(config: Config, key=None):
        """ Create a new gene with values initialized by config. """
        return NodeGene(key=next(config.node_key_counter) if key is None else key,
                        response=config.response_config.new(),
                        bias=config.bias_config.new(),
                        activation=config.activation_config.new(),
                        aggregation=config.aggregation_config.new())

    @staticmethod
    def crossover(node1, node2):
        """ Create a new gene randomly inheriting attributes from its parents. """
        assert node1.key == node2.key, "You can only crossover matching/homologous genes"
        return NodeGene(key=node1.key,
                        response=node1.response if random.random() < .5 else node2.response,
                        bias=node1.bias if random.random() < .5 else node2.bias,
                        activation=node1.activation if random.random() < .5 else node2.activation,
                        aggregation=node1.aggregation if random.random() < .5 else node2.aggregation)

    def mutate(self, config):
        self.response = config.response_config.mutate(self.response)
        self.bias = config.bias_config.mutate(self.bias)
        self.activation = config.activation_config.mutate(self.activation)
        self.aggregation = config.aggregation_config.mutate(self.aggregation)

    def copy(self):
        return NodeGene(self.key, self.response, self.bias, self.activation, self.aggregation)

    @staticmethod
    def distance(node1, node2, config):
        d = abs(node1.bias - node2.bias) + abs(node1.response - node2.response)
        if node1.activation != node2.activation:
            d += 1.0
        if node1.aggregation != node2.aggregation:
            d += 1.0
        return d * config.compatibility_weight_coefficient


@dataclass
class ConnectionGene:
    """
    Base class for CPPN connection genes
    """
    key: tuple  # (input, output) primary key
    weight: float
    enabled: bool

    def disable(self):
        self.enabled = False

    @staticmethod
    def new(config: Config, key, weight=None):
        """ Create a new gene with values initialized by config. """
        return ConnectionGene(key=key,
                              weight=config.weight_config.new() if weight is None else weight,
                              enabled=config.enabled_config.new())

    @staticmethod
    def crossover(conn1, conn2):
        """ Create a new gene randomly inheriting attributes from its parents. """
        assert conn1.key == conn2.key, "You can only crossover matching/homologous genes"
        return ConnectionGene(key=conn1.key,
                              weight=conn1.weight if random.random() < .5 else conn2.weight,
                              enabled=conn1.enabled if random.random() < .5 else conn2.enabled)

    def mutate(self, config):
        self.weight = config.weight_config.mutate(self.weight)
        self.enabled = config.enabled_config.mutate(self.enabled)

    def copy(self):
        return ConnectionGene(self.key, self.weight, self.enabled)

    @staticmethod
    def distance(conn1, conn2, config):
        d = abs(conn1.weight - conn2.weight)
        if conn1.enabled != conn2.enabled:
            d += 1.0
        return d * config.compatibility_weight_coefficient
