"""
Defines blueprint for configuration and functions applied to species.
"""

from itertools import count
from dataclasses import dataclass, field
from typing import *

from neat.model import *
from neat.util.funcs import stat_functions


# --------------- POPULATION/SPECIES CONFIGURABLES ---------------

@dataclass
class SpeciesBP:
    """ Blueprint for species. """

    # Dynamic compatibility threshold
    compat_threshold_initial: float  # initial compatibility threshold value
    compat_threshold_modifier: float  # amount by which to adjust the compat threshold if num of species is off-target
    compat_threshold_min: float # minimum value of compatibility threshold
    target_num_species: int  # dynamic compatibility threshold used to maintain this target

    # Stagnation
    species_fitness_func: str  # how to measure fitness of a species based on its members' fitness (for stagnation)
    max_stagnation: int  # how long before a species can be removed for not improving its species-fitness (15)
    species_elitism: int  # number of species with highest species-fitness are protected from stagnation
    reset_on_extinction: bool  # init new population if all species simultaneously become extinct due to stagnation

    # Species ID counter
    __id_counter: count = field(default_factory=count)

    def create(self, **kwargs):
        """ Create a new Species with the given attributes. """

        if "id" not in kwargs:
            kwargs["id"] = next(self.__id_counter)

        return Species(**kwargs)
    
    def get_species_fitness(self, species: Species):
        """ Get the species fitness function. """
        return stat_functions[self.species_fitness_func](species.get_fitnesses())
