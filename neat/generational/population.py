""" Implements the core NEAT evolution algorithm. """

from neat.base import BasePopulation, BaseSpeciesSet, BaseSpecies, BaseAgent, BaseGenome
from .config import Config


class Population(BasePopulation):
    """
    A group of individuals capable of evolving
    """
    def __init__(self, config: Config, species_set_type=BaseSpeciesSet, species_type=BaseSpecies,
                 agent_type=BaseAgent, genome_type=BaseGenome):
        super().__init__(config, species_set_type=species_set_type, species_type=species_type,
                         agent_type=agent_type, genome_type=genome_type)

    def evaluate(self, fitness_function=None):
        self.species_set.speciate(self.agents)

        # Evaluate individuals and assign score
        for agent in self.agents.values():
            # If no fitness function provided, do not assign fitness
            if fitness_function:
                agent.fitness = fitness_function(agent)
            if self.fittest is None or agent.fitness > self.fittest.fitness:
                self.fittest = agent

        next_gen_genomes = []

        # Put best genomes from each species into next generation
        for species in self.species_set.index.values():
            fittest_in_species = max(species.members, key=lambda m: m.fitness)
            next_gen_genomes.append(fittest_in_species.genome)

        # Breed the rest of the genomes
        while len(next_gen_genomes) < self.config.pop_size:
            # Choose a species probabilistically based on average fitness
            parent_species = self.species_set.random_species()
            # Choose two parent agents from that species probabilistically based on fitness
            parent1, parent2 = parent_species.random_members(2)

            # Crossover genomes
            if parent1.fitness < parent2.fitness:  # parent1 must be fitter parent
                parent2, parent1 = parent1, parent2
            child_genome = self.genome_type(next(self.gid_indexer), self.config)
            child_genome.crossover(parent1.genome, parent2.genome)

            # Mutate genome
            child_genome.mutate()

            next_gen_genomes.append(child_genome)

        return next_gen_genomes

