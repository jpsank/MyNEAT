""" Demonstrate functionality of evolution algorithm without phenome generation. """

from neat.realtime.population import Agent, Population
from neat.realtime.config import Config


config = Config()
population = Population(config, Agent)
population.init()


def evaluate_genome(agent):
    """
    Evaluate genome on its ability to maximize number of connections per number of nodes
    Max number of connections, where nodes can connect to themselves, given n nodes = n(n-1)/2 + n
    """
    num_nodes = len(agent.genome.nodes)
    max_conns = num_nodes*(num_nodes-1)/2 + num_nodes  # maximum possible number of connections given this many nodes

    num_conns = len(agent.genome.connections)
    return num_conns / max_conns


gen = 0
while True:
    for gid, agent in population.agents.items():
        agent.fitness = evaluate_genome(agent)

    population.update()

    print(f"Fittest conns: {len(population.fittest.genome.connections)}, "
          f"enabled: {len([c for c in population.fittest.genome.connections.values() if c.enabled])}, "
          f"nodes: {len(population.fittest.genome.nodes)}, Species: {len(population.species_set.dict)}, "
          f"Pop: {len(population.agents)}, ")
    print("\t" + ", ".join(f"{k[0]}->{k[1]}" for k in population.fittest.genome.connections.keys()))
    gen += 1
