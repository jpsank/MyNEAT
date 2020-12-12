""" Divides the population into species based on genomic distances. """

import random
from itertools import count

from myneat.genome import Genome
from myneat.config import Config


class GenomeDistanceCache:
    def __init__(self, config: Config):
        self.distances = {}
        self.config = config
        self.hits = 0
        self.misses = 0

    def __call__(self, genome1, genome2):
        """ Return distance between two genomes """
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
    def __init__(self, sid, mascot, ticks):
        self.id = sid
        self.created = ticks
        self.last_improved = ticks
        self.members = set()

        self.add(mascot)
        self.mascot = mascot

    def add(self, m):
        """ Add new member """
        self.members.add(m)
        m.species = self

    def fitness(self):
        """ Return average fitness of members """
        return sum(m.fitness for m in self.members) / len(self.members)

    def reset(self):
        """ Remove all members except mascot """
        for member in self.members:
            member.species = None
        self.members = set()
        self.add(self.mascot)

    def random_members(self, k=1):
        """ Return k random members chosen probabilistically based on fitness """
        return random.choices(self.members, weights=[m.fitness for m in self.members], k=k)

    def size(self):
        """ Return size of species """
        return len(self.members)


class SpeciesSet:
    def __init__(self, config: Config):
        self.config = config
        self.species_dict: {int: Species} = {}
        self.indexer = count()

        self.compat_threshold = config.compat_threshold_initial

    def new_species(self, mascot):
        """ Create an entirely new species with given mascot """
        species = Species(next(self.indexer), mascot)
        self.species_dict[species.id] = species
        return species

    def adjust_compat_threshold(self):
        """ Adjust dynamic compatibility threshold to better fit target number of species """
        num_species = len(self.species_dict)
        if num_species > self.config.target_num_species:
            self.compat_threshold += self.config.compat_threshold_modifier
        elif num_species < self.config.target_num_species:
            self.compat_threshold -= self.config.compat_threshold_modifier

        if self.compat_threshold < self.config.compat_threshold_min:
            self.compat_threshold = self.config.compat_threshold_min

    def reset(self):
        """ Reset all species but preserve mascots """
        for species in self.species_dict.values():
            species.reset()

    def speciate(self, agents):
        """ Speciate agents """

        # Reset all species but preserve mascots
        self.reset()

        # Assign each agent's species
        distances = GenomeDistanceCache(self.config)
        for agent in agents.values():
            # Skip if agent is a mascot (to ensure mascots are not reassigned to another species)
            if agent.species is not None:
                continue

            # If compatibility distance < threshold, individual belongs to this species
            found = False
            for (sid, species) in self.species_dict.items():
                if distances(agent.genome, species.mascot.genome) < self.compat_threshold:
                    species.add(agent)
                    found = True
                    break

            # If not compatible with any species, create new species and assign as mascot
            if not found:
                self.new_species(agent)

    def remove_empty(self):
        """ Remove empty species """
        for sid, species in self.species_dict.items():
            if species.size() == 0:
                del self.species_dict[sid]

    def random_species(self, k=1):
        """ Choose species probabilistically based on average fitness """
        all_species = list(self.species_dict.values())
        return random.choices(all_species, weights=[s.fitness() for s in all_species], k=k)

