from dataclasses import dataclass


class Gene:
    """ Abstract base class for all genes. """


@dataclass
class NodeGene(Gene):
    """
    Base class for CPPN node genes.
    out = activation(bias + response * aggregation(inputs))
    """
    id: int
    response: float
    bias: float
    activation: str
    aggregation: str


@dataclass
class ConnGene(Gene):
    """ Base class for CPPN connection genes. """
    in_node: int
    out_node: int
    weight: float
    enabled: bool

    @property
    def key(self):
        return (self.in_node, self.out_node)
