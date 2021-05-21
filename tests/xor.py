import random
import time

from neat.base.config import StringConfig
from neat.generational import Config, BaseAgent, Population
from neat.base.nn import FeedForwardNetwork


class Agent(BaseAgent):
    def __init__(self, genome):
        super().__init__(genome)
        self.brain = FeedForwardNetwork.create(genome)


def eval_fitness(agent):
    """
    Evaluates fitness of the agent
    Arguments:
        agent: The agent to evaluate
    Returns:
        The fitness score - the higher score the means the better
        fit organism. Maximal score: 16.0
    """
    xor_inputs = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
    xor_outputs = [(0.0,), (1.0,), (1.0,), (0.0,)]
    error_sum = 0.0
    for xi, xo in zip(xor_inputs, xor_outputs):
        output = agent.brain.activate(xi)
        error_sum += abs(output[0] - xo[0])

    fitness = (4 - error_sum) ** 2
    return fitness


class Xor:
    def __init__(self, cfg):
        self.population = Population(cfg, agent_type=Agent)

    def run(self, fitness_threshold=15.5, max_generations=None, verbose=False):
        try:
            next_gen_genomes = None
            g = 0
            while True:
                if verbose:
                    print("Generation", g + 1)
                self.population.init(next_gen_genomes)
                next_gen_genomes = self.population.evaluate(eval_fitness)

                if verbose:
                    print("Best fitness:", self.population.fittest.fitness)
                    # fits = [a.fitness for a in self.population.agents.values()]
                    # print("Average fitness:", sum(fits) / len(fits))
                    # print()

                if fitness_threshold and self.population.fittest.fitness >= fitness_threshold:  # 1.1  # 1.166666
                    print(f"Success after {g + 1} generation(s).")
                    break
                if max_generations and g + 1 >= max_generations:
                    print(f"Failure after {g + 1} generation(s).")
                    break
                g += 1
        except KeyboardInterrupt:
            print("Stopped.")

    def test(self):
        print("Testing fittest solution:")
        xor_inputs = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
        xor_outputs = [(0.0,), (1.0,), (1.0,), (0.0,)]
        for xi, xo in zip(xor_inputs, xor_outputs):
            output = self.population.fittest.brain.activate(xi)
            error = abs(output[0] - xo[0])
            print(f"{xi} => {xo}:", round(output[0], 3), "error:", round(error, 3))


if __name__ == '__main__':
    config = Config(num_inputs=2, num_outputs=1)
                    # pop_size=150, target_num_species=80,
                    # species_elitism=2, elitism=2, max_stagnation=20, reset_on_extinction=False,
                    # compat_threshold_modifier=0.0)
    config.activation_config = StringConfig(
        default="sigmoid",
        mutate_rate=0.0,
        options=["sigmoid"]
    )
    config.aggregation_config = StringConfig(
        default="sum",
        mutate_rate=0.0,
        options=["sum"]
    )

    # trials_duration = 0
    # epochs_duration = 0
    # generations_per_trial = 0
    for t in range(20):
        print(f"Trial #{t+1}")
        timer = time.time()
        xor = Xor(config)
        xor.run(max_generations=20000, verbose=True)
        timer = time.time() - timer
        print(f"Trial duration: {timer}")
        xor.test()
        print()


