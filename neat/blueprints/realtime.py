"""
Defines blueprint for a real-time NEAT simulation.
"""

from dataclasses import dataclass
from typing import *

from neat.model import *
from neat.blueprints.population import PopulationBP


# --------------- SIMULATION CONFIGURABLES ---------------

@dataclass
class RealtimeBP:
    """
    High-level controls for a rt-NEAT (realtime NEAT) simulation.

    Based on law of eligibility: n = m/(P*I)
        n: replacement frequency (number of ticks between replacements)
        m: minimum age (minimum time alive, in ticks, before an agent is eligible to be removed)
        P: population size
        I: preferred ineligibility fraction (fraction of population at any given time that should be ineligible)
    """

    # Population config
    population: PopulationBP

    # Realtime parameters
    minimum_age: int  # The minimum time alive, in ticks, before an agent is eligible to be removed
    ineligibility_fraction: float  # The preferred fraction of population at any given time that should be ineligible
    reorganization_frequency: int  # Adjust compat threshold & reassign species every _ replacements (=5 in NERO)
    replacement_frequency: int = None  # The number of ticks between replacements

    def __post_init__(self):
        if self.replacement_frequency is None:
            self.replacement_frequency = round(self.minimum_age / (self.population.pop_size * self.ineligibility_fraction))

    def do_replacement(self, population: Population):
        """ Replace one eligible bad agent with the offspring of two good agents """

        # Sort agents with age >= minimum_age by fitness in ascending order
        eligible_agents = [a for a in population.agents.values() if a.age >= self.minimum_age]
        eligible_agents = sorted(eligible_agents, key=lambda a: a.fitness)
        # NOTE: The above used to use adjusted fitness rather than raw fitness. Should it be adjusted fitness?
        
        # Cancel when no agents are eligible to be removed
        if len(eligible_agents) == 0:
            return

        # Remove agent with age >= minimum_age and lowest fitness
        worst = eligible_agents[0]
        del population.agents[worst.genome.id]

        # Select two parents (parent 1 is fitter than parent 2) probabilistically weighted by fitness
        parent_species = population.get_a_random_species(weighted=True)
        parent1, parent2 = parent_species.get_random_members(k=2, weighted=True)
        if parent1.fitness < parent2.fitness:
            parent2, parent1 = parent1, parent2
        
        # Crossover and mutate parent genomes to create child genome
        child = self.population.genome.crossover(parent1.genome, parent2.genome)
        self.population.genome.mutate(child)

        # Create offspring agent and set species
        agent = Agent(genome=child)
        parent_species.add(agent)
        population.agents[child.id] = agent

    def do_reorganization(self, population: Population):
        """ Reorganize agents into species using dynamic compatibility threshold """

        # Adjust dynamic compatibility threshold
        self.population.adjust_compat_threshold(population)

        # Reset all species by removing members
        # Then, for each agent (who is not a mascot),
        #   assign to first species whose mascot is compatible;
        #   otherwise, assign as mascot to new species
        self.population.speciate(population)

        # Remove any empty species (cleanup routine)
        # After reassigning, some empty species may be left, so delete them
        population.remove_empty_species()

    def update(self, population: Population):
        """
        Call every tick of simulation. 
        Assumes agents have already been evaluated and fitness assigned.
        """

        population.fittest = max(population.agents.values(), key=lambda a: a.fitness)

        # Stagnation step
        self.population.check_stagnation(population)

        # Check for complete extinction
        if self.population.species.reset_on_extinction and len(population.agents) == 0:
            self.population.reset(population)
            print("Reset on total extinction")

        # Replace (reproduction step)
        if population.ticks % self.replacement_frequency == 0:
            self.do_replacement(population)
            print("Replacement")

            # Reorganization (speciation step)
            if population.replacements % self.reorganization_frequency == 0:
                self.do_reorganization(population)
                print("Reorganization")

            population.replacements += 1

        population.ticks += 1

