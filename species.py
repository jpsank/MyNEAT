""" Divides the population into species based on genomic distances. """

import random

from myneat.genome import Genome
from myneat.config import Config


class GenomeDistanceCache:
    def __init__(self, config: Config):
        self.distances = {}
        self.config = config
        self.hits = 0
        self.misses = 0

    def __call__(self, genome1, genome2):
        distance = self.distances.get((genome1.id, genome2.id))
        if distance is None:
            # Distance is not already computed.
            distance = Genome.distance(genome1, genome2, self.config)
            self.distances[genome1.id, genome2.id] = distance
            self.distances[genome2.id, genome1.id] = distance
            self.misses += 1
        else:
            self.hits += 1
        return distance


class Species:
    def __init__(self, sid, mascot):
        self.id = sid
        self.mascot = mascot
        self.members = [self.mascot]
        self.total_fitness = 0

    @staticmethod
    def new(config, mascot):
        return Species(config.species_id_counter(), mascot)

    def reset(self):
        self.mascot = random.choice(self.members)
        self.members = [self.mascot]
        self.total_fitness = 0


class SpeciesSet:
    def __init__(self, config: Config):
        self.config = config
        self.species: {int: Species} = {}

    def add_species(self, species):
        self.species[species.id] = species

    def create_species(self, mascot):
        new_species = Species.new(self.config, mascot)
        self.add_species(new_species)
        return new_species

    def speciate(self, individuals):
        distances = GenomeDistanceCache(self.config)

        for species in self.species.values():
            species.reset()

        for individual in individuals:
            found_species = False
            for (sid, species) in self.species.items():
                if distances(individual.genome, species.mascot.genome) < self.config.compatibility_threshold:
                    # compatibility distance is less than threshold, so individual belongs to this species
                    species.members.append(individual)
                    individual.species = species
                    found_species = True
                    break
            if not found_species:
                new_species = self.create_species(individual)
                individual.species = new_species

        # Remove unused species
        for (sid, species) in self.species.items():
            if len(species.members) == 0:
                del self.species[sid]


