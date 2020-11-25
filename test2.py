""" Demonstrate functionality of evolution algorithm without phenome generation. """

from myneat.population import Population
from myneat.genome import Genome
from myneat.config import default_config


population = Population(default_config)
population.init()


def evaluate_genome(individual):
    """
    Evaluate genome on its ability to maximize number of connections per number of nodes
    Max number of connections, where nodes can connect to themselves, given n nodes = n(n-1)/2 + n
    """
    num_nodes = len(individual.genome.nodes)
    max_conns = num_nodes*(num_nodes-1)/2 + num_nodes  # maximum possible number of connections given this many nodes

    num_conns = len(individual.genome.connections)
    return num_conns / max_conns


gen = 0
while True:
    population.evaluate(evaluate_genome)
    print(f"Fittest conns: {len(population.fittest.genome.connections)}, "
          f"enabled: {len([c for c in population.fittest.genome.connections.values() if c.enabled])}, "
          f"nodes: {len(population.fittest.genome.nodes)}, Species: {len(population.species_set.species)}, "
          f"Pop: {len(population.individuals)}, ")
    print("\t" + ", ".join(f"{k[0]}->{k[1]}" for k in population.fittest.genome.connections.keys()))
    gen += 1
