from neat.generational import *

import cProfile
import gym
import longdpole
import numpy as np


class Agent(BaseAgent):
    def __init__(self, genome):
        super().__init__(genome)
        self.brain = RecurrentNetwork.new(genome)


class Game:
    def __init__(self):
        self.env = gym.make('LongdpoleEnv-v0')
        self.env.seed(123)
        print("Gym initialized")
        cfg = Config(num_inputs=3, num_outputs=2,
                     pop_size=500, target_num_species=200, max_stagnation=15,
                     compatibility_disjoint_coefficient=1, compatibility_weight_coefficient=3,
                     node_add_prob=0.03, node_delete_prob=0.03,
                     conn_add_prob=0.3, conn_delete_prob=0.3)
        self.population = Population(cfg, agent_type=Agent)
        self.population.init()
        print("Population initialized")

    def run_single(self, nn, n_episodes=100000, render=False):
        obs = self.env.reset()
        fitness = 0
        for _ in range(n_episodes):
            if render:
                self.env.render()
            output = nn.activate(obs)
            # action = int(output[1] > output[0])  # 1 (right) if second output > first output, else 0 (left)
            obs, reward, done, _ = self.env.step(output)
            fitness += reward
            if done:
                break
        return fitness

    def calculate_fitness(self, agent):
        return self.run_single(agent.brain)

    def run(self, render_fittest=True):
        i = 0
        while True:
            print("Generation", i+1)
            next_gen_genomes = self.population.evaluate(self.calculate_fitness)

            fits = [a.fitness for a in self.population.agents.values()]
            print("Median fitness:", sorted(fits)[len(fits) // 2])
            print("Average fitness:", sum(fits)/len(fits))
            print("Best fitness:", self.population.fittest.fitness)

            if render_fittest:
                print("Rendering fittest")
                self.env.viewer = None
                self.run_single(self.population.fittest.brain, n_episodes=300, render=True)
                self.env.close()  # Close the viewer
            print()

            self.population.init(next_gen_genomes)
            i += 1


def run():
    game = Game()
    game.run()


if __name__ == '__main__':
    run()
    # cProfile.run("run()", sort='tottime')
