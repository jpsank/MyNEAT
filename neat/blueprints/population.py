"""
Defines blueprint for configuration and functions applied to populations.
"""

from dataclasses import dataclass
from typing import *

from neat.model import *
from neat.blueprints.genome import GenomeBP
from neat.blueprints.species import SpeciesBP

@dataclass
class PopulationBP:
    """ Blueprint for populations. """

    # General parameters
    pop_size: int  # The number of individuals in the population

    # Genome config
    genome: GenomeBP

    # Species config
    species: SpeciesBP

    def create_new_agents(self) -> Dict[int, Agent]:
        """ Create a new map of randomly initialized agents. """
        agents = {}
        for _ in range(self.pop_size):
            genome = self.genome.create()
            agents[genome.id] = Agent(genome=genome)
        return agents

    def create(self) -> Population:
        """ Create a new population. """
        pop = Population(
            agents=self.create_new_agents(), 
            compat_threshold=self.species.compat_threshold_initial)
        self.speciate(pop)
        return pop
    
    def reset(self, population: Population):
        """ Reset a population. """
        population.agents = self.create_new_agents()
        population.species = {}
        population.compat_threshold = self.species.compat_threshold_initial
        self.speciate(population)

    def __get_distance_with_cache(self, a: Genome, b: Genome, cache: dict) -> float:
        """ Get the distance between two genomes, using cache for efficiency. """
        dist = cache.get((a.id, b.id))
        if dist is None:
            # Distance is not already computed.
            dist = self.genome.distance(a, b)
            cache[a.id, b.id] = dist
            cache[b.id, a.id] = dist
        return dist

    def speciate(self, population: Population, new_mascots=True):
        """ Assign agents in a population to species based on genomic distance. """

        # Create a cache for efficiency
        distance_cache = {}

        if new_mascots:
            # If mascots are old (from the last generation), find the best mascot for each existing species.
            for species in population.species.values():
                # The new mascot is the genome closest to the current mascot.
                new_mascot = min(
                    population.agents.values(), 
                    key=lambda a: self.__get_distance_with_cache(species.mascot.genome, a.genome, cache=distance_cache))
                species.reset(new_mascot)
        else:
            # Reset all species, preserving mascots
            population.reset_all_species()

        # Assign each agent's species
        for agent in population.agents.values():
            # Skip if agent is a mascot (to ensure mascots are not reassigned to another species)
            if agent.species_id is not None:
                continue

            # If compatibility distance < threshold, individual belongs to this species
            found = False
            for s in population.species.values():
                dist = self.__get_distance_with_cache(agent.genome, s.mascot.genome, cache=distance_cache)
                if dist < population.compat_threshold:
                    s.add(agent)
                    found = True
                    break

            # If not compatible with any species, create new species and assign as mascot
            if not found:
                s = self.species.create(mascot=agent, created_at=population.ticks)
                population.species[s.id] = s
    
    def check_stagnation(self, population: Population):
        """ Check if any species has not improved in a while. If so, remove them. """

        # Update species' fitness history
        for species in population.species.values():
            species.fitness = self.species.get_species_fitness(species)
            species.fitness_history.append(species.fitness)
            if species.best_fitness is None or species.fitness > species.best_fitness:
                species.last_improved = population.ticks
                species.best_fitness = species.fitness

        # Sort by ascending fitness
        all_species = list(population.species.values())
        all_species.sort(key=lambda s: s.fitness)

        # Save best species if elitism is enabled
        if self.species.species_elitism > 0:
            all_species = all_species[:-self.species.species_elitism]
        
        # Remove species if they have not improved in a while
        for species in all_species:
            if species.last_improved is not None and population.ticks - species.last_improved >= self.species.max_stagnation:
                population.remove_species(species.id)

    def adjust_compat_threshold(self, population: Population):
        """ Adjust dynamic compatibility threshold to better fit target number of species. """
        diff = len(population.species) - self.species.target_num_species
        population.compat_threshold += (diff / self.pop_size) * self.species.compat_threshold_modifier

        # Ensure compatibility threshold is within bounds
        population.compat_threshold = max(population.compat_threshold, self.species.compat_threshold_min)
