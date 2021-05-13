""" Implements the core evolution algorithm. """

from neat.base.genome import BaseGenome
from neat.base.species import BaseSpeciesSet, BaseSpecies
from neat.base.data import Index

from itertools import count


class BaseAgent:
    def __init__(self, genome):
        self.genome = genome
        self.fitness = 0
        self.species = None
        self.brain = None  # neural net
        self.age = 0

    @property
    def id(self): return self.genome.id

    def adjusted_fitness(self):
        return self.fitness / len(self.species.members)


class BasePopulation:
    """
    A group of individuals, capable of evolving
    """
    def __init__(self, config, species_set_type=BaseSpeciesSet, species_type=BaseSpecies,
                 agent_type=BaseAgent, genome_type=BaseGenome):
        self.config = config
        self.species_set = species_set_type(config, species_type)
        self.agent_type = agent_type
        self.agents = Index("id")

        self.genome_type = genome_type
        self.gid_indexer = count()
        self.fittest = None
        self.ticks = 0

    def new_genome(self):
        genome = self.genome_type(next(self.gid_indexer), self.config)
        genome.init()
        return genome

    def init(self, genomes=None):
        """ Initialize agents and species """
        self.agents.clear()
        if genomes is None:
            for _ in range(self.config.pop_size):
                self.agents.add(self.agent_type(self.new_genome()))
        else:
            for genome in genomes:
                self.agents.add(self.agent_type(genome))
        self.species_set.speciate(self.agents, self.ticks)

    # SPECIES STUFF

    def remove_species(self, sid):
        species = self.species_set.index[sid]
        for member in species.members:
            self.agents.remove(member)
        del self.species_set.index[sid]

    def do_stagnation(self):
        # Update species' best average fitness
        all_species = self.species_set.index.values()
        for species in all_species:
            fitness = self.config.species_fitness_func(species.get_fitnesses())
            if species.best_fitness is None or fitness > species.best_fitness:
                species.best_fitness = fitness
                species.last_improved = self.ticks

        # Sort by best fitness in ascending order
        all_species = sorted(all_species, key=lambda s: s.best_fitness)

        # Check stagnation
        num_remaining = len(all_species)
        for species in all_species:
            # Override stagnation if removing this species would
            # result in the total number of species dropping below the limit.
            # Because species are in ascending fitness order, less fit species will be removed first.
            if num_remaining > self.config.species_elitism:
                stagnant_time = self.ticks - species.last_improved
                if stagnant_time > self.config.max_stagnation:
                    self.remove_species(species.id)
                    num_remaining -= 1

