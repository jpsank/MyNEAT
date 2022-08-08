from typing import *
from dataclasses import dataclass

from neat.model.genome import Genome

@dataclass
class Agent:
    genome: Genome
    species_id: int = None
    fitness: float = None
    age: int = 0