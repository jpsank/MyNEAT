"""
Defines blueprint for a generational NEAT simulation.
"""

from dataclasses import dataclass
import random
from typing import *
import math

from neat.model import *
from neat.blueprints.population import PopulationBP


# --------------- SIMULATION CONFIGURABLES ---------------

class TotalExtinctionException(Exception):
    pass


@dataclass
class GenerationalBP:
    """ High-level controls for a generational NEAT simulation """

    # Population config
    population: PopulationBP

    # Generational reproduction parameters
    elitism: int  # The number of most-fit individuals in each species to be preserved as-is from one generation to next
    survival_threshold: float  # The fraction of members for each species allowed to reproduce each generation
    min_species_size: int  # The minimum number of genomes per species after reproduction

    def evaluate(self, population: Population, fitness_func):
        """ Evaluate the fitness of all agents in the population. """
        
        # Evaluate agents and assign fitness scores
        fittest, least_fit = None, None
        for agent in population.agents.values():
            agent.fitness = fitness_func(agent)

            # Keep track of max and min fitness
            if fittest is None or agent.fitness > fittest.fitness:
                fittest = agent
            if least_fit is None or agent.fitness < least_fit.fitness:
                least_fit = agent
        
        population.fittest = fittest
        population.least_fit = least_fit

        # Compute species' adjusted fitnesses
        # Do not allow the fitness range to be zero, as we divide by it below.
        # TODO: The ``1.0`` below is rather arbitrary, and should be configurable.
        fitness_range = max(1.0, fittest.fitness - least_fit.fitness)
        for species in population.species.values():
            msf = sum(species.get_fitnesses()) / species.size()
            af = (msf - least_fit.fitness) / fitness_range
            species.adjusted_fitness = af

    def compute_spawn(self, population: Population):
        """
        Compute the proper number of offspring per species (proportional to fitness).
        Pre-condition: species' adjusted fitnesses must be populated.
        TODO: Implementation should be made more readable.
        """

        # Effective min_species_size is max(min_species_size, elitism)
        min_species_size = max(self.min_species_size, self.elitism)
        af_sum = sum([s.adjusted_fitness for s in population.species.values()])
        spawn_amounts = {}
        for species in population.species.values():
            s = min_species_size
            if af_sum > 0:
                s = max(s, species.adjusted_fitness / af_sum * self.population.pop_size)

            d = (s - species.size()) * 0.5
            c = int(round(d))
            spawn = species.size()
            if abs(c) > 0:
                spawn += c
            elif d > 0:
                spawn += 1
            elif d < 0:
                spawn -= 1

            spawn_amounts[species.id] = spawn

        # Normalize the spawn amounts so that the next generation is roughly
        # the population size requested by the user.
        total_spawn = sum(spawn_amounts.values())
        norm = self.population.pop_size / total_spawn
        spawn_amounts = {
            sid: max(self.min_species_size, int(round(n * norm))) for sid, n in spawn_amounts.items()}

        return spawn_amounts

    def reproduce(self, population: Population):
        """
        Reproduce the next generation of agents.
        Pre-condition: agents' fitnesses and species' adjusted fitnesses must be populated.
        """

        # Compute spawn amounts
        spawn_amounts = self.compute_spawn(population)

        # Reproduction step
        next_gen_agents = {}
        for sid, spawn in spawn_amounts.items():
            species = population.species[sid]
            assert spawn > 0

            # Sort members in order of descending fitness.
            old_members = species.members
            species.members.sort(reverse=True, key=lambda a: a.fitness)

            # Transfer elites to new generation.
            if self.elitism > 0:
                for m in old_members[:self.elitism]:
                    next_gen_agents[m.genome.id] = m
                    spawn -= 1

            if spawn <= 0:
                continue

            # Only use the survival threshold fraction to use as parents for the next generation.
            repro_cutoff = int(math.ceil(self.survival_threshold * len(old_members)))
            # Use at least two parents no matter what the threshold fraction result is.
            repro_cutoff = max(repro_cutoff, 2)
            old_members = old_members[:repro_cutoff]

            # Randomly choose parents and produce the number of offspring allotted to the species.
            while spawn > 0:
                spawn -= 1

                parent1 = random.choice(old_members)
                parent2 = random.choice(old_members)

                # Parent 1 must be fitter parent
                if parent1.fitness < parent2.fitness:
                    parent2, parent1 = parent1, parent2

                # Note that if the parents are not distinct, crossover will produce a
                # genetically identical clone of the parent (but with a different ID).
                new_genome = self.population.genome.crossover(parent1.genome, parent2.genome)
                self.population.genome.mutate(new_genome)
                next_gen_agents[new_genome.id] = Agent(genome=new_genome)

                population.ancestors[new_genome.id] = (parent1.genome.id, parent2.genome.id)
        
        # Replace old agents with new agents
        population.agents = next_gen_agents

        # OLD CODE:
        # next_gen_agents = {}
        # for s in population.species.values():
        #     # Randomize species' mascot
        #     s.mascot = s.random_member()
            
        #     # Put best genome from each species into next generation
        #     best_member = s.get_best()
        #     next_gen_agents[best_member.genome.id] = best_member

        # # Breed the rest of the genomes
        # while len(next_gen_agents) < self.population.pop_size:
        #     # Choose a species probabilistically based on average fitness
        #     parent_species = population.random_species()

        #     # Choose two parent agents from that species probabilistically based on fitness
        #     a, b = parent_species.random_members(k=2)

        #     # Parent A must be fitter parent
        #     if a.fitness < b.fitness:
        #         b, a = a, b
            
        #     # Crossover and mutate to create child genome
        #     c = self.population.genome.crossover(a.genome, b.genome)
        #     self.population.genome.mutate(c)

        #     # Add to next generation
        #     next_gen_agents[c.id] = Agent(genome=c)

    def next_generation(self, population: Population):
        """ Create the next generation of agents and their species. """

        # Stagnation step
        self.population.check_stagnation(population)

        # Check for complete extinction
        if len(population.species) == 0:
            if self.population.species.reset_on_extinction:
                self.population.reset(population)
                print("Reset on total extinction")
            else:
                raise TotalExtinctionException()
        
        # Reproduce next generation
        self.reproduce(population)
        
        # Adjust dynamic compatibility threshold
        self.population.adjust_compat_threshold(population)

        # Speciate agents, assigning new mascots to existing species
        self.population.speciate(population, new_mascots=True)
        
        population.ticks += 1
    
    def run(self, population: Population, fitness_func=None, max_generations=20000, fitness_threshold=None):
        """ Run a generational NEAT simulation. """
        g = 1
        while g <= max_generations and (fitness_threshold is None or population.fittest.fitness < fitness_threshold):
            self.evaluate(population, fitness_func=fitness_func)
            self.next_generation(population)
            g += 1
