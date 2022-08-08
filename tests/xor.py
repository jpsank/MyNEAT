import random
import time

from neat.model import Agent, Population
from neat.blueprints import *
from neat.nn import FeedForwardNetwork
from neat.util.vis import *


xor_bp = GenerationalBP(

    # Population blueprint
    population = PopulationBP(
        
        # General parameters
        pop_size = 150,

        # Species blueprint
        species = SpeciesBP(
            # Dynamic compatibility threshold
            compat_threshold_initial = 3.0,
            compat_threshold_modifier = 0.1,
            compat_threshold_min = 0.1,
            target_num_species = 50,

            # Stagnation
            species_fitness_func = "mean",
            max_stagnation = 20,
            species_elitism = 2,
            reset_on_extinction = False,
        ),

        # Genome blueprint
        genome = GenomeBP(

            # Node options
            node = NodeBP(
                activation = StringBP(
                    default="sigmoid",
                    mutate_rate=0.0,
                    options=["sigmoid"]
                ),
                aggregation = StringBP(
                    default="sum",
                    mutate_rate=0.0,
                    options=["sum"]
                ),
                bias = FloatBP(
                    init_mean=0.0,
                    init_stdev=1.0,
                    max_value=30.0,
                    min_value=-30.0,
                    mutate_power=0.5,
                    mutate_rate=0.7,
                    replace_rate=0.1
                ),
                response = FloatBP(
                    init_mean=1.0,
                    init_stdev=0.0,
                    max_value=30.0,
                    min_value=-30.0,
                    mutate_power=0.0,
                    mutate_rate=0.0,
                    replace_rate=0.0
                ),
            ),

            # Connection options
            conn = ConnBP(
                enabled = BoolBP(
                    default=True,
                    mutate_rate=0.01
                ),
                weight = FloatBP(
                    init_mean=0.0,
                    init_stdev=1.0,
                    max_value=30.0,
                    min_value=-30.0,
                    mutate_power=0.5,
                    mutate_rate=0.8,
                    replace_rate=0.1
                ),
            ),

            # Network initialization options
            num_inputs = 2,
            num_outputs = 1,

            # Network mutation options
            conn_add_prob = 0.5,
            conn_delete_prob = 0.5,
            node_add_prob = 0.2,
            node_delete_prob = 0.2,

            # Genome compatibility options
            compatibility_disjoint_coefficient = 1.0,
            compatibility_weight_coefficient = 0.5,

            # Structural mutations
            single_structural_mutation = False,
            structural_mutation_surer = False,
        ),

    ),

)


XOR_INPUTS = ((0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0))
XOR_OUTPUTS = ((0.0,), (1.0,), (1.0,), (0.0,))


def eval_fitness(agent: Agent):
    """
    Evaluates fitness of the agent
    Arguments:
        agent: The agent to evaluate
    Returns:
        The fitness score - the higher score the means the better
        fit organism. Maximal score: 4.0
    """
    brain = FeedForwardNetwork.create(agent.genome, xor_bp.population.genome.input_ids, xor_bp.population.genome.output_ids)
    fitness = 4.0
    for xi, xo in zip(XOR_INPUTS, XOR_OUTPUTS):
        output = brain.activate(xi)
        fitness -= (output[0] - xo[0]) ** 2
    return fitness


def run_trial(population: Population, fitness_threshold=3.9, max_generations=None, verbose=False, plot_genomes=False):
    while True:
        if verbose: print("Generation", population.ticks)

        xor_bp.evaluate(population, eval_fitness)

        if verbose:
            print("Best fitness:", population.fittest.fitness)
            # fits = [a.fitness for a in population.agents.values()]
            # print("Average fitness:", sum(fits) / len(fits))
            # print()

        if plot_genomes and population.ticks % 10 == 0:
            # print([len(s.members) for s in population.species.values()])
            best_agents = [max(s.members, key=lambda a: a.fitness) for s in population.species.values()]
            best_agents = sorted(best_agents, key=lambda a: a.fitness, reverse=True)
            plt_genomes({a.species_id: a.genome for a in best_agents}, xor_bp.population.genome.input_ids, xor_bp.population.genome.output_ids)

        if fitness_threshold and population.fittest.fitness >= fitness_threshold:
            print(f"Success after {population.ticks} generation(s).")
            break
        if max_generations and population.ticks >= max_generations:
            print(f"Failure after {population.ticks} generation(s).")
            break

        xor_bp.next_generation(population)

def test(population: Population):
    print("Testing fittest solution:")
    brain = FeedForwardNetwork.create(population.fittest.genome, xor_bp.population.genome.input_ids, xor_bp.population.genome.output_ids)
    for xi, xo in zip(XOR_INPUTS, XOR_OUTPUTS):
        output = brain.activate(xi)
        error = abs(output[0] - xo[0])
        print(f"{xi} => {xo}:", round(output[0], 3), "error:", round(error, 3))


if __name__ == '__main__':
    num_trials = 1  # default 50
    fitness_threshold = 3.9  # default 3.9
    verbose = True  # default False
    plot_genomes = True  # default False

    for t in range(num_trials):
        print(f"Trial #{t+1}")
        start_time = time.time()

        # Create population and run
        population = xor_bp.population.create()
        try:
            run_trial(population, max_generations=20000, fitness_threshold=fitness_threshold, verbose=verbose, plot_genomes=plot_genomes)
        except KeyboardInterrupt:
            print("Stopped.")
        
        print(f"Trial duration: {time.time() - start_time}s")
        
        test(population)
        print()


