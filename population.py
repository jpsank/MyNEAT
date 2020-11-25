""" Implements the core evolution algorithm. TODO: Add real-time evolution. """

import random

from myneat.genome import Genome
from myneat.config import Config
from myneat.species import SpeciesSet


class Individual:
    def __init__(self, genome):
        self.genome = genome
        self.fitness = 0
        self.species = None


class Population:
    """
    A group of genetically homologous individuals, capable of evolving
    """
    def __init__(self, config: Config, individual_type=Individual):
        self.config = config
        self.species_set = SpeciesSet(self.config)
        self.individual_type = individual_type

        self.individuals = {}
        self.fittest = None

    def init(self, genomes=None):
        if genomes is None:
            for _ in range(self.config.pop_size):
                g = Genome.new(self.config)
                g.init()
                self.individuals[g.id] = self.individual_type(g)
        else:
            self.individuals = {genome.id: self.individual_type(genome) for genome in genomes}

    def evaluate(self, evaluate_fitness):
        self.species_set.speciate(self.individuals.values())

        # Evaluate individuals and assign score
        for individual in self.individuals.values():
            fitness = evaluate_fitness(individual)
            adjusted_fitness = fitness / len(individual.species.members)
            individual.fitness = adjusted_fitness
            individual.species.total_fitness += individual.fitness
            if self.fittest is None or fitness > self.fittest.fitness:
                self.fittest = individual

        next_gen_genomes = []

        # Put best genomes from each species into next generation
        for species in self.species_set.species.values():
            sorted_members = sorted(species.members, key=lambda m: m.fitness, reverse=True)
            fittest_in_species = sorted_members[0]
            next_gen_genomes.append(fittest_in_species.genome)

        # Breed the rest of the genomes
        while len(next_gen_genomes) < self.config.pop_size:
            species = random.choices(list(self.species_set.species.values()),
                                     weights=[s.total_fitness for s in self.species_set.species.values()], k=1)[0]
            parent1, parent2 = random.choices(species.members,
                                              weights=[m.fitness for m in species.members], k=2)
            # Parent1 must be fitter parent
            if parent1.fitness < parent2.fitness:
                parent2, parent1 = parent1, parent2
            child_genome = Genome.crossover(parent1.genome, parent2.genome)
            child_genome.mutate()
            next_gen_genomes.append(child_genome)

        self.init(next_gen_genomes)

