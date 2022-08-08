import time
import gym
import cProfile
import numpy as np

from neat.blueprints import *
from neat.model import Population
from neat.nn import FeedForwardNetwork
import neat.util.vis as vis


bp = GenerationalBP(

    # Reproduction parameters
    elitism = 5,
    survival_threshold = 0.5,
    min_species_size = 2,

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
            target_num_species = 10,

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
            num_inputs = 8,
            num_outputs = 4,

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


class Game:
    def __init__(self, env_id):
        self.env_id = env_id
        self.env = gym.make(env_id)
        self.env.reset()
    
    def obs_to_input(self, obs):
        """ 
        Convert cart-pole observation to input for the neural network.
            obs = [position of cart, velocity of cart, angle of pole, rotation rate of pole].
        Defined at https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py#L75 
        """
        if self.env_id.startswith('CartPole'):
            return obs[0], *obs[2:]  # Remove the velocity input
        else:
            return obs

    def output_to_action(self, output):
        """ Convert neural network output to action for the cart-pole simulation. """
        return np.argmax(output)

    def run_simulation(self, nn, n_episodes=100000, render=False):
        # if render:
        #     self.env.viewer = None
        obs = self.env.reset()
        fitness = 0
        for _ in range(n_episodes):
            inputs = self.obs_to_input(obs)
            output = nn.activate(inputs)
            action = self.output_to_action(output)

            obs, reward, done, info = self.env.step(action)
            fitness += reward

            if render:
                self.env.render()
            if done:
                break
        # if render:
            # self.env.close()  # Close the viewer
        return fitness

    def eval_fitness(self, agent, **kwargs):
        brain = FeedForwardNetwork.create(agent.genome, bp.population.genome.input_ids, bp.population.genome.output_ids)
        return self.run_simulation(brain, **kwargs)


def run_trial(population: Population, game: Game, fitness_threshold=100000, max_generations=None, render_fittest=True, verbose=False, plot_genomes=False):
    while True:
        if verbose:
            print("Generation", population.ticks)

        bp.evaluate(population, game.eval_fitness)

        if verbose:
            print("Best fitness:", population.fittest.fitness, "Avg:", population.get_average_fitness())
            print("Species:", {s.id: s.size() for s in population.species.values()})
            print()

        if plot_genomes:
            vis.plt_population(population, bp)

        if render_fittest:
            game.eval_fitness(population.fittest, n_episodes=300, render=True)

        if fitness_threshold and population.fittest.fitness >= fitness_threshold:
            print(f"Success after {population.ticks} generation(s).")
            break
        if max_generations and population.ticks >= max_generations:
            print(f"Failure after {population.ticks} generation(s).")
            break

        bp.next_generation(population)


def run():
    num_trials = 1  # default 50
    fitness_threshold = 200  # default 200
    verbose = True  # default False
    render_fittest = True  # default False
    plot_genomes = True  # default False
    env_id = "LunarLander-v2"  # "CartPole-v0" or "LongdpoleEnv-v0"

    print("Initializing gym...")
    game = Game(env_id)

    total_generations = 0
    for t in range(num_trials):
        print(f"Trial #{t+1}")
        start_time = time.time()

        # Create population and run
        population = bp.population.create()
        try:
            run_trial(population, game, fitness_threshold=fitness_threshold, max_generations=20000, render_fittest=render_fittest, verbose=verbose, plot_genomes=plot_genomes)
        except KeyboardInterrupt:
            print("Stopped.")
        
        total_generations += population.ticks
        
        print(f"Trial duration: {time.time() - start_time}s")
        print()
    
    avg_generations = total_generations / num_trials
    print(f"Average generations per trial over {num_trials} trial(s):", avg_generations)


if __name__ == '__main__':
    run()
    # cProfile.run("run()", sort='tottime')
