from typing import *
import random
from dataclasses import dataclass, field

from neat.model.agent import Agent
from neat.model.species import Species


@dataclass
class Population:
    """ A population of agents. """
    
    agents: 'dict[int, Agent]'
    compat_threshold: float
    species: 'dict[int, Species]' = field(default_factory=dict)

    # Statistics

    ticks: int = 0
    replacements: int = 0
    ancestors: 'dict[int, list]' = field(default_factory=dict)
    fittest: Agent = None
    least_fit: Agent = None

    # Accessors

    def get_ancestors(self, agent_id):
        """ Get the ancestors of an agent. """
        return self.ancestors.get(agent_id, [])
    
    def get_random_species(self, k=1, weighted=False) -> 'list[Species]':
        """ Return species chosen randomly or probabilistically weighted by adjusted fitness. """
        species_list = list(self.species.values())
        if weighted:
            choices = random.choices(species_list, weights=[s.adjusted_fitness for s in species_list], k=k)
        else:
            choices = random.sample(species_list, k=k)
        return choices
    
    def get_a_random_species(self, weighted=False) -> Species:
        """ Choose a single species randomly or probabilistically weighted by adjusted fitness. """
        return self.get_random_species(k=1, weighted=weighted)[0]
    
    def get_average_fitness(self) -> float:
        """ Return average fitness of all agents. """
        return sum([a.fitness for a in self.agents.values()]) / len(self.agents)

    # Mutators

    def remove_species(self, sid):
        """ Remove species with given id. """
        species = self.species.pop(sid)
        for a_id in species.members:
            self.agents.pop(a_id)
    
    def remove_empty_species(self):
        """ Remove all empty species. """
        for sid, s in self.species.items():
            if s.is_empty():
                del self.species[sid]
    
    def reset_all_species(self):
        """ Reset all species, preserving their mascots. """
        for species in self.species.values():
            species.reset()
    

