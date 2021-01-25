""" Implements the core evolution algorithm. """

from myneat.myneat.genome import Genome
from myneat.myneat.config import Config
from myneat.myneat.species import SpeciesSet


class Agent:
    def __init__(self, genome):
        self.genome = genome
        self.fitness = 0
        self.species = None
        self.brain = None  # neural net
        self.age = 0

    def adjusted_fitness(self):
        return self.fitness / len(self.species.members)


class Population:
    """
    A group of individuals capable of evolving
    """
    def __init__(self, config: Config, agent_type):
        self.config = config
        self.species_set = SpeciesSet(config)

        self.agent_type = agent_type
        self.agents: {int: Agent} = {}
        self.fittest = None

        self.ticks = 0
        self.replacements = 0

    def new_agent(self, genome=None):
        if genome is None:
            genome = Genome.new(self.config)
            genome.init()
        a = self.agent_type(genome)
        self.agents[genome.id] = a
        return a

    def init(self, genomes=None):
        """ Initialize agents and species """
        if genomes is None:
            for _ in range(self.config.pop_size):
                self.new_agent()
        else:
            for genome in genomes:
                self.new_agent(genome)
        self.species_set.speciate(self.agents, self.ticks)

    def do_replacement(self):
        """ Replace one bad agent with offspring of some good agents """

        # Sort agents with age >= minimum_age by adjusted fitness in ascending order
        eligible_agents = sorted([gid for (gid, a) in self.agents.items() if a.age >= self.config.minimum_age],
                                 key=lambda a: a.adjusted_fitness())
        # Cancel when no agents are eligible to be removed
        if len(eligible_agents) == 0:
            return

        # Remove agent with age >= minimum_age and lowest adjusted fitness
        worst = eligible_agents[0]
        del self.agents[worst]

        # Choose a parent species probabilistically based on average fitness
        parent_species = self.species_set.random_species()

        # Choose two members probabilistically based on fitness; these are the parents
        parent1, parent2 = parent_species.random_members(2)

        # Crossover genomes
        if parent1.fitness < parent2.fitness:  # parent1 must be fitter parent
            parent2, parent1 = parent1, parent2
        child_genome = Genome.crossover(parent1.genome, parent2.genome)

        # Mutate genome
        child_genome.mutate()

        # Create offspring agent and set species
        a = self.new_agent(child_genome)
        parent_species.add(a)

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
        s = self.species_set.species_dict[sid]
        for member in s.members:
            gid = member.genome.id
            del self.agents[gid]
        del self.species_set.species_dict[sid]

    def do_stagnation(self):
        # Sort species in ascending fitness order
        species_items = self.species_set.species_dict.items()

        # Update species' best fitness
        for sid, species in species_items:
            fitness = species.fitness()
            if species.best_fitness is None or fitness > species.best_fitness:
                species.best_fitness = fitness
                species.last_improved = self.ticks

        # Sort by best fitness
        species_items = sorted(self.species_set.species_dict.items(), key=lambda item: item[1].best_fitness)

        # Check stagnation
        num_remaining = len(species_items)
        for sid, species in species_items:
            # Override stagnation if removing this species would
            # result in the total number of species dropping below the limit.
            # Because species are in ascending fitness order, less fit species will be removed first.
            if num_remaining > self.config.species_elitism:
                stagnant_time = self.ticks - species.last_improved
                if stagnant_time > self.config.max_stagnation:
                    self.remove_species(sid)
                    num_remaining -= 1

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

