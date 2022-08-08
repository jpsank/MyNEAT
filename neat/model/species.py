from typing import *
import random
from dataclasses import dataclass, field

from neat.model import Agent

@dataclass
class Species:
    id: int
    mascot: Agent
    members: 'list[Agent]' = field(default_factory=list)

    # Statistics
    
    created_at: int = None
    last_improved: int = None

    fitness: float = None
    best_fitness: float = None
    fitness_history: 'list[float]' = field(default_factory=list)

    adjusted_fitness: float = None

    def __post_init__(self):
        self.add(self.mascot)

    # Accessors

    def get_fitnesses(self) -> 'list[float]':
        """ Return fitnesses of members. """
        return [m.fitness for m in self.members]

    def get_random_members(self, k=1, weighted=False) -> 'list[Agent]':
        """ Return k members chosen randomly or probabilistically based on fitness. """
        if weighted:
            return random.choices(self.members, weights=self.get_fitnesses(), k=k)
        else:
            return random.sample(self.members, k=k)

    def get_random_member(self, weighted=False) -> Agent:
        """ Return member chosen randomly or probabilistically based on fitness. """
        return self.get_random_members(1, weighted=weighted)[0]

    def get_best(self) -> Agent:
        """ Return best member. """
        return max(self.members, key=lambda m: m.fitness)
    
    def size(self) -> int:
        """ Returns number of members. """
        return len(self.members)
    
    def is_empty(self) -> bool:
        """ Returns True if no members. """
        return self.size() == 0

    # Mutators

    def add(self, agent: Agent):
        """ Add new member. """
        assert agent.species_id is None, "Agent already belongs to a species"

        agent.species_id = self.id
        if agent not in self.members:
            self.members.append(agent)
    
    def remove(self, agent: Agent):
        """ Remove a member. """
        assert agent is not self.mascot, "Cannot remove mascot"

        agent.species_id = None
        self.members.remove(agent)

    def reset(self, new_mascot: Agent = None):
        """ Remove all members except mascot. Or, set new mascot if given. """
        for m in self.members:
            m.species_id = None
        self.members.clear()
        if new_mascot is None:
            self.add(self.mascot)
        else:
            self.set_mascot(new_mascot)

    def set_mascot(self, agent: Agent):
        """ Set new mascot. """
        assert agent.species_id == None or agent.species_id == self.id, "Agent belongs to another species"
        
        if agent.species_id == None:
            self.add(agent)
        
        self.mascot = agent
    
    # def set_fitness(self, fitness: float, ticks: int):
    #     """ Set fitness of species. """
    #     self.fitness = fitness
    #     self.fitness_history.append(fitness)
    #     if self.best_fitness is None or fitness > self.best_fitness:
    #         self.best_fitness = fitness
    #         self.last_improved = ticks
