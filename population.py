""" Implements the core evolution algorithm. TODO: Add real-time evolution. """

import random
from itertools import count

from myneat.genome import Genome
from myneat.config import Config
from myneat.species import SpeciesSet


class Agent:
    def __init__(self, genome):
        self.genome = genome
        self.fitness = 0
        self.species = None
        self.brain = None  # neural net
        self.age = 0

    def adjusted_fitness(self):
        return self.fitness / len(self.species.members)


class Population:
    """
    A group of genetically homologous individuals, capable of evolving
    """
    def __init__(self, config: Config):
        self.config = config
        self.species_set = SpeciesSet(config)

        self.agents = {}
        self.fittest = None

        self.ticks = 0
        self.replacements = 0

    def new_agent(self, genome):
        a = Agent(genome)
        self.agents[genome.id] = a
        return a

    def init(self, genomes=None):
        if genomes is None:
            for _ in range(self.config.pop_size):
                g = Genome.new(self.config)
                g.init()
                self.agents[g.id] = Agent(g)
        else:
            self.agents = {genome.id: Agent(genome) for genome in genomes}

    def do_replacement(self):
        """ Replace one bad agent with offspring of some good agents """

        # Remove agent with age >= minimum_age and lowest adjusted fitness
        eligible_agents = sorted([gid for (gid, a) in self.agents.items() if a.age >= self.config.minimum_age],
                                 key=lambda a: a.adjusted_fitness())
        worst = eligible_agents[0]
        del self.agents[worst]

        # Choose a parent species probabilistically based on average fitness
        parent_species = self.species_set.random_species()

        # Choose two members probabilistically based on fitness; these are the parents
        parent1, parent2 = parent_species.random_members(2)

        # Crossover genomes
        if parent1.fitness < parent2.fitness:  # parent1 must be fitter parent
            parent2, parent1 = parent1, parent2
        child_genome = Genome.crossover(parent1.genome, parent2.genome)

        # Mutate genome
        child_genome.mutate()

        # Create offspring agent and set species
        a = self.new_agent(child_genome)
        parent_species.add(a)

    def do_reorganization(self):
        """ Reorganize species using dynamic compatibility threshold """

        # Adjust dynamic compatibility threshold
        self.species_set.adjust_compat_threshold()

        # Reset all species by removing members
        # Then, for each agent (who is not a mascot),
        #   assign to first species whose mascot is compatible;
        #   otherwise, assign as mascot to new species
        self.species_set.speciate(self.agents)

        # Remove any empty species (cleanup routine)
        # After reassigning, some empty species may be left, so delete them
        self.species_set.remove_empty()

    def update(self):
        """ Call every tick of simulation. Assumes agents have already been evaluated and fitness assigned """

        # Replace (reproduction step)
        if self.ticks % self.config.replacement_frequency == 0:
            self.do_replacement()
            self.replacements += 1

        # TODO: Stagnation step

        # Check for complete extinction
        if self.config.reset_on_extinction and len(self.agents) == 0:
            self.init()

        # Reorganization (speciation step)
        if self.replacements % self.config.reorganization_frequency:
            self.do_reorganization()

        self.ticks += 1


if __name__ == '__main__':
    config = Config()
    pop = Population(config)
    pop.init()

    i = 0
    while i < 500:

        # Evaluate agents
        for agent in pop.agents.values():
            agent.fitness = agent.genome.size()  # Example evaluation

        # Update population
        pop.update()

        i += 1
