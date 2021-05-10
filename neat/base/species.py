""" Divides the population into species based on genomic distances. """

import random
from itertools import count

from neat.base.data import Index


class GenomeDistanceCache:
    def __init__(self, config):
        self.distances = {}
        self.config = config
        self.hits = 0
        self.misses = 0

    def __call__(self, genome1, genome2):
        """ Return distance between two genomes. Genomes must have the same type. """
        assert (genome_type := type(genome1)) == type(genome2)

        distance = self.distances.get((genome1.id, genome2.id))
        if distance is None:
            # Distance is not already computed.
            distance = genome_type.distance(genome1, genome2, self.config)
            self.distances[genome1.id, genome2.id] = distance
            self.distances[genome2.id, genome1.id] = distance
            self.misses += 1
        else:
            self.hits += 1
        return distance


class BaseSpecies:
    def __init__(self, sid, mascot, ticks=None):
        self.id = sid
        self.best_fitness = None
        self.members = set()

        self.created = ticks
        self.last_improved = ticks

        self.add(mascot)
        self.mascot = mascot

    def add(self, m):
        """ Add new member """
        self.members.add(m)
        m.species = self

    def avg_fitness(self):
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
        return random.choices(list(self.members), weights=[m.fitness for m in self.members], k=k)


class BaseSpeciesSet:
    def __init__(self, config, species_type=BaseSpecies):
        self.config = config
        self.species_type = species_type
        self.index = Index("id")
        self.sid_indexer = count()

        self.compat_threshold = config.compat_threshold_initial

    def new(self, mascot, ticks=None):
        """ Create and add a new species with given mascot. """
        species = self.species_type(next(self.sid_indexer), mascot, ticks=ticks)
        self.index.add(species)
        return species

    def adjust_compat_threshold(self):
        """ Adjust dynamic compatibility threshold to better fit target number of species. """
        diff = len(self.index) - self.config.target_num_species
        self.compat_threshold += (diff / self.config.pop_size) * self.config.compat_threshold_modifier

        if self.compat_threshold < self.config.compat_threshold_min:
            self.compat_threshold = self.config.compat_threshold_min

    def reset_all(self):
        """ Reset all species, preserving their mascots. """
        for species in self.index.values():
            species.reset()

    def speciate(self, agents_index, ticks=None):
        """ Speciate agents. """
        # Reset all species but preserve mascots
        self.reset_all()

        # Assign each agent's species
        distances = GenomeDistanceCache(self.config)
        for agent in agents_index.values():
            # Skip if agent is a mascot (to ensure mascots are not reassigned to another species)
            if agent.species is not None:
                continue

            # If compatibility distance < threshold, individual belongs to this species
            found = False
            for (sid, species) in self.index.items():
                if distances(agent.genome, species.mascot.genome) < self.compat_threshold:
                    species.add(agent)
                    found = True
                    break

            # If not compatible with any species, create new species and assign as mascot
            if not found:
                self.new(agent, ticks)

    def remove_empty(self):
        """ Remove empty species """
        for sid, species in self.index.items():
            if len(species.members) == 0:
                del self.index[sid]

    def random_species(self, k=1):
        """ Choose species probabilistically based on average fitness """
        all_species = list(self.index.values())
        choices = random.choices(all_species, weights=[s.avg_fitness() for s in all_species], k=k)
        if k == 1:
            return choices[0]
        return choices

    def size(self):
        return len(self.index)

