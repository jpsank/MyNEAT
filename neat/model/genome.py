from typing import *
import random
from dataclasses import dataclass, field

from neat.model.genes import NodeGene, ConnGene


@dataclass
class Genome:
    """ Base class for genomes. """
    id: int
    nodes: 'dict[int, NodeGene]'
    conns: 'dict[tuple(int, int), ConnGene]'
    
    def size(self) -> int:
        """ Returns genome 'complexity', taken to be number of nodes + number of connections. """
        return len(self.nodes) + len(self.conns)
