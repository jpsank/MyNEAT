""" Implements the core rt-NEAT evolution algorithm. """

from neat.base.genome import BaseGenome
from neat.base.species import BaseSpeciesSet, BaseSpecies
from neat.base.population import BasePopulation, BaseAgent
from neat.realtime.config import Config


class Species(BaseSpecies):
    pass


class SpeciesSet(BaseSpeciesSet):
    pass


class Agent(BaseAgent):
    pass


class Genome(BaseGenome):
    pass


class Population(BasePopulation):
    """
    A group of individuals capable of evolving
    """
    def __init__(self, config: Config, species_set_type=SpeciesSet, species_type=Species,
                 agent_type=Agent, genome_type=Genome):
        super().__init__(config, species_set_type=species_set_type, species_type=species_type,
                         agent_type=agent_type, genome_type=genome_type)
        self.replacements = 0

    def do_replacement(self):
        """ Replace one bad agent with offspring of some good agents """

        # Sort agents with age >= minimum_age by adjusted fitness in ascending order
        eligible_agents = sorted(filter(lambda a: a.age >= self.config.minimum_age, self.agents.values()),
                                 key=lambda a: a.adjusted_fitness())
        # Cancel when no agents are eligible to be removed
        if len(eligible_agents) == 0:
            return

        # Remove agent with age >= minimum_age and lowest adjusted fitness
        worst = eligible_agents[0]
        self.agents.remove(worst)

        # Choose a parent species probabilistically based on average fitness
        parent_species = self.species_set.random_species()

        # Choose two members probabilistically based on fitness; these are the parents
        parent1, parent2 = parent_species.random_members(2)

        # Crossover genomes
        if parent1.fitness < parent2.fitness:  # parent1 must be fitter parent
            parent2, parent1 = parent1, parent2
        child_genome = Genome(next(self.gid_indexer), self.config)
        child_genome.crossover(parent1.genome, parent2.genome)

        # Mutate genome
        child_genome.mutate()

        # Create offspring agent and set species
        child = self.agent_type(child_genome)
        self.agents.add(child)
        parent_species.add(child)

    def do_reorganization(self):
        """ Reorganize species using dynamic compatibility threshold """

        # Adjust dynamic compatibility threshold
        self.species_set.adjust_compat_threshold()

        # Reset all species by removing members
        # Then, for each agent (who is not a mascot),
        #   assign to first species whose mascot is compatible;
        #   otherwise, assign as mascot to new species
        self.species_set.speciate(self.agents, self.ticks)

        # Remove any empty species (cleanup routine)
        # After reassigning, some empty species may be left, so delete them
        self.species_set.remove_empty()

    def remove_species(self, sid):
        s = self.species_set.dict[sid]
        for member in s.members:
            self.agents.remove(member)
        del self.species_set.dict[sid]

    def update(self):
        """ Call every tick of simulation. Assumes agents have already been evaluated and fitness assigned """

        self.fittest = max(self.agents.values(), key=lambda a: a.fitness)

        # Stagnation step
        self.do_stagnation()

        # Check for complete extinction
        if self.config.reset_on_extinction and len(self.agents) == 0:
            self.init()
            print("Reset on total extinction")

        # Replace (reproduction step)
        if self.ticks % self.config.replacement_frequency == 0:
            self.do_replacement()
            print("Replacement")

            # Reorganization (speciation step)
            if self.replacements % self.config.reorganization_frequency == 0:
                self.do_reorganization()
                print("Reorganization")

            self.replacements += 1

        self.ticks += 1

