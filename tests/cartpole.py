from neat.base.nn import FeedForwardNetwork
from neat.generational import *

import cProfile
import gym
import longdpole
import numpy as np


class Agent(BaseAgent):
    def __init__(self, genome):
        super().__init__(genome)
        self.brain = FeedForwardNetwork.create(genome)


class Game:
    def __init__(self, cfg):
        self.env = None
        self.population = Population(cfg, agent_type=Agent)

    def init(self, seed=1):
        print("Seeding gym...")
        self.env.seed(seed)
        print("Initializing population...")
        self.population.init()

    def observation_to_input(self, obs):
        """ Convert cart-pole observation to input for the neural network. """
        return obs

    def output_to_action(self, output):
        """ Convert neural network output to action for the cart-pole simulation. """
        return output

    def run_one(self, nn, n_episodes=100000, render=False):
        obs = self.env.reset()
        fitness = 0
        for _ in range(n_episodes):
            if render:
                self.env.render()

            inputs = self.observation_to_input(obs)
            output = nn.activate(inputs)
            action = self.output_to_action(output)

            obs, reward, done, _ = self.env.step(action)
            fitness += reward
            if done:
                break
        return fitness

    def calculate_fitness(self, agent):
        return self.run_one(agent.brain)

    def run(self, fitness_threshold=100000, max_generations=None, render_fittest=True, verbose=False):
        g = 0
        while True:
            if verbose:
                print("Generation", g+1)
            next_gen_genomes = self.population.evaluate(self.calculate_fitness)

            if verbose:
                fits = [a.fitness for a in self.population.agents.values()]
                print("Median fitness:", sorted(fits)[len(fits) // 2])
                print("Average fitness:", sum(fits)/len(fits))
                print("Best fitness:", self.population.fittest.fitness)

            if render_fittest:
                print("Rendering fittest")
                self.env.viewer = None
                self.run_one(self.population.fittest.brain, n_episodes=300, render=True)
                self.env.close()  # Close the viewer

            if fitness_threshold and self.population.fittest.fitness >= fitness_threshold:  # 1.1  # 1.166666
                print(f"Success after {g + 1} generation(s).")
                break
            if max_generations and g + 1 >= max_generations:
                print(f"Failure after {g + 1} generation(s).")
                break

            self.population.init(next_gen_genomes)
            g += 1
        return g + 1


class SingleGame(Game):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.env = gym.make('CartPole-v0')  # .env gets rid of TimeLimit wrapper

    def observation_to_input(self, obs):
        """ obs = [position of cart, velocity of cart, angle of pole, rotation rate of pole].
        Defined at https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py#L75 """
        # return obs
        return obs[0], *obs[2:]  # Remove the velocity input

    def output_to_action(self, output):
        return int(output[1] > output[0])  # Go right (1) if second output > first output, else go left (0)


class DoubleGame(Game):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.env = gym.make('LongdpoleEnv-v0')


def run():
    print("Initializing game...")
    cfg = Config(num_inputs=3, num_outputs=2)
                 # pop_size=150, max_stagnation=15,
                 # compatibility_disjoint_coefficient=1, compatibility_weight_coefficient=3,
                 # node_add_prob=0.03, node_delete_prob=0.03,
                 # conn_add_prob=0.3, conn_delete_prob=0.3,
                 # compat_threshold_modifier=0.0)
    game = SingleGame(cfg)
    game.init()

    total_generations = 0
    num_trials = 50
    for t in range(num_trials):
        print(f"Trial #{t+1}")
        total_generations += game.run(fitness_threshold=200, render_fittest=False)

    avg_generations = total_generations / num_trials
    print(f"Average generations per trial over {num_trials} trials:", avg_generations)


if __name__ == '__main__':
    run()
    # cProfile.run("run()", sort='tottime')
