
from neat.generational import Config, BaseAgent, Population
from neat.base.nn import RecurrentNetwork


class Agent(BaseAgent):
    def __init__(self, genome):
        super().__init__(genome)
        self.brain = RecurrentNetwork.new(genome)


def calculate_errors(nn):
    train_x = ((0, 0), (1, 1), (1, 0), (0, 1))
    train_y = (0, 0, 1, 1)
    for inputs, output in zip(train_x, train_y):
        yield abs(output - nn.activate(inputs)[0])


def calculate_fitness(agent):
    total_error = sum(calculate_errors(agent.brain))

    score = 1 / (1+total_error)

    if score >= 0.99999:
        score += 1 / agent.genome.size()

    return score


if __name__ == '__main__':
    config = Config(num_inputs=2, num_outputs=1, pop_size=100, target_num_species=40)
    population = Population(config, agent_type=Agent)

    next_gen_genomes = None
    i = 0
    while True:
        print("Generation", i+1)
        population.init(next_gen_genomes)
        next_gen_genomes = population.evaluate(calculate_fitness)

        print("Best fitness:", population.fittest.fitness)
        fits = [a.fitness for a in population.agents.values()]
        print("Average fitness:", sum(fits)/len(fits))
        print()

        if population.fittest.fitness >= 1.16666:
            print("Success")
            break
        i += 1

    print("Fittest solution:")
    print("Nodes:", len(population.fittest.genome.nodes), population.fittest.genome.nodes)
    print("Conns:", len(population.fittest.genome.connections), population.fittest.genome.connections)
    print("TESTS")
    print("(0, 0) => 0:", population.fittest.brain.activate([0, 0]))
    print("(1, 1) => 0:", population.fittest.brain.activate([1, 1]))
    print("(1, 0) => 1:", population.fittest.brain.activate([1, 0]))
    print("(0, 1) => 1:", population.fittest.brain.activate([0, 1]))

