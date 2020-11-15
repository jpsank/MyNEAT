""" Demonstrate functionality of evolution algorithm without phenome generation. """

from myneat.population import Population
from myneat.genome import Genome
from myneat.config import default_config


population = Population(default_config)

# TODO: make it easier to initialize. How do other implementations do this?
genomes = []
genome = Genome.new(default_config)
genome.init()
for _ in range(default_config.pop_size):
    genomes.append(genome.copy())
population.init(genomes)


def evaluate_genome(individual):
    """ Evaluate genome on its ability to maximize number of connections n(n-1)/2 per number of nodes """
    num_nodes = len(individual.genome.nodes)
    max_conns = num_nodes*(num_nodes-1)/2  # compute maximum possible number of connections given this many nodes
    return len(individual.genome.connections) - max_conns


gen = 0
while gen < 500:
    population.evaluate(evaluate_genome)
    print(f"Fittest conns: {len(population.fittest.genome.connections)}, "
          f"enabled: {len([c for c in population.fittest.genome.connections.values() if c.enabled])}, "
          f"nodes: {len(population.fittest.genome.nodes)}, Species: {len(population.species_set.species)}, "
          f"Pop: {len(population.individuals)}, ")
    print(population.fittest.genome.connections)
    gen += 1
