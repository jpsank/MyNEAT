""" Implements the core NEAT evolution algorithm. """

from neat.base.population import BasePopulation, BaseAgent
from neat.base.genome import BaseGenome
from neat.base.species import BaseSpeciesSet, BaseSpecies
from neat.generational.config import Config


class Population(BasePopulation):
    """
    A group of individuals capable of evolving
    """
    def __init__(self, config: Config, species_set_type=BaseSpeciesSet, species_type=BaseSpecies,
                 agent_type=BaseAgent, genome_type=BaseGenome):
        super().__init__(config, species_set_type=species_set_type, species_type=species_type,
                         agent_type=agent_type, genome_type=genome_type)

    def evaluate(self, evaluate_fitness):
        self.species_set.speciate(self.agents)

        # Evaluate individuals and assign score
        for agent in self.agents.values():
            agent.fitness = evaluate_fitness(agent)
            if self.fittest is None or agent.adjusted_fitness() > self.fittest.fitness:
                self.fittest = agent

        next_gen_genomes = []

        # Put best genomes from each species into next generation
        for species in self.species_set.index.values():
            sorted_members = sorted(species.members, key=lambda m: m.fitness, reverse=True)
            fittest_in_species = sorted_members[0]
            next_gen_genomes.append(fittest_in_species.genome)

        # Breed the rest of the genomes
        while len(next_gen_genomes) < self.config.pop_size:
            # Choose a species probabilistically based on average fitness
            species = self.species_set.random_species(1)
            # Choose two parent agents from that species probabilistically based on fitness
            parent1, parent2 = species.random_members(2)

            # Crossover genomes
            if parent1.fitness < parent2.fitness:  # parent1 must be fitter parent
                parent2, parent1 = parent1, parent2
            child_genome = self.genome_type(next(self.gid_indexer), self.config)
            child_genome.crossover(parent1.genome, parent2.genome)

            # Mutate genome
            child_genome.mutate()

            next_gen_genomes.append(child_genome)

        self.init(next_gen_genomes)

