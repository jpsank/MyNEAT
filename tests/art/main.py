from neat.base.nn import FeedForwardNetwork
from neat.generational import BaseAgent, Population, Config

import math
import pickle
import pygame
import numpy as np
from numpy import sin, cos
import os


# CONFIG
# Determine path to configuration file. This path manipulation is
# here so that the script will run successfully regardless of the
# current working directory.
local_dir = os.path.dirname(__file__)
save_dir = os.path.join(local_dir, "save")
data_dir = os.path.join(local_dir, "data")

IMG_WIDTH = 160
IMG_HEIGHT = 160

UPSCALE = 1


# COLORS
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
YELLOW = (255, 255, 0)

# PYGAME SETUP
pygame.font.init()  # init font
STAT_FONT = pygame.font.SysFont("comicsans", 40)


def close_factors(num):
    t = int(math.sqrt(num))
    while num % t != 0:
        t -= 1
    return t, int(num / t)


def clip(n, low, high):
    return max(low, min(high, n))


class ArtistAgent(BaseAgent):
    def __init__(self, genome):
        super().__init__(genome)
        self.brain = FeedForwardNetwork.create(genome)
        self.Z = None

    def generate(self, width=IMG_WIDTH, height=IMG_HEIGHT):
        self.Z = np.empty((height, width, 3))
        for y in range(height):
            for x in range(width):
                new_x, new_y = math.pi * (2 * (x/width) - 1), math.pi * (2 * (y/height) - 1)
                dist = math.sqrt(new_x**2 + new_y**2)
                # self.Z[y][x] = self.net.activate((new_x, new_y, dist, sin(new_x), cos(new_x), sin(new_y), cos(new_y)))
                self.Z[y][x] = self.brain.activate((new_x, new_y, dist, cos(new_x), sin(new_y)))
        self.Z = 255 * (self.Z + 1) / 2

    def draw(self, win, x, y):
        if UPSCALE == 1:
            surf = pygame.surfarray.make_surface(self.Z)
            win.blit(surf, (x, y))
        else:
            orig_x = x
            for row in self.Z:
                for col, val in enumerate(row):
                    pygame.draw.rect(win, val, (x, y, UPSCALE, UPSCALE))
                    x += UPSCALE
                x = orig_x
                y += UPSCALE

    def draw_selection(self, win, x, y, thick=1):
        height, width, _ = self.Z.shape
        pygame.draw.rect(win, YELLOW, (x, y, width*UPSCALE, height*UPSCALE), thick)


class Simulation:
    def __init__(self, cfg):
        self.width = None
        self.height = None
        self.win = None

        self.generation = 0
        self.population = Population(cfg, agent_type=ArtistAgent)
        self.population.init()

    def run_generation(self, save=True):
        factors = close_factors(len(self.population.agents))
        n_rows, n_cols = min(factors), max(factors)

        if self.win is None:
            self.width = n_cols * IMG_WIDTH * UPSCALE
            self.height = n_rows * IMG_HEIGHT * UPSCALE
            self.win = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("Art Evolution")

        coord_to_agent = {}
        col, row = 0, 0
        for agent in self.population.agents.values():
            agent.generate()
            agent.draw(self.win, x=col * IMG_WIDTH * UPSCALE, y=row * IMG_HEIGHT * UPSCALE)
            coord_to_agent[(row, col)] = agent
            col += 1
            if col >= n_cols:
                col = 0
                row += 1

        clock = pygame.time.Clock()

        # generations label
        gen_label = STAT_FONT.render(str(self.generation + 1), True, (255, 255, 255))
        self.win.blit(gen_label, (5, 5))

        done = False
        selected = set()
        while not done:
            clock.tick(30)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                    pygame.quit()
                    quit()
                    break
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RETURN and len(selected) > 0:
                        done = True
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    pos_x, pos_y = pygame.mouse.get_pos()
                    col = int(pos_x / IMG_WIDTH / UPSCALE)
                    row = int(pos_y / IMG_HEIGHT / UPSCALE)
                    x, y = col * IMG_WIDTH * UPSCALE, row * IMG_HEIGHT * UPSCALE

                    agent = coord_to_agent[(row, col)]
                    agent.fitness += 1
                    agent.draw_selection(self.win, x, y, thick=agent.fitness)
                    selected.add(agent)

            pygame.display.update()

        if save:
            run = 0
            while os.path.exists(os.path.join(save_dir, f"run{run}-gen{self.generation}-winner.pickle")):
                run += 1
            for n, winner in enumerate(selected):
                path = os.path.join(save_dir, f"run{run}-gen{self.generation}-winner{'' if n==0 else n+1}.pickle")
                pickle.dump(winner.brain, open(path, "wb"))

    def run(self):
        """ Runs the NEAT algorithm to train neural networks to make art. """

        try:
            while True:
                self.run_generation()
                next_gen_genomes = self.population.evaluate()
                self.population.init(next_gen_genomes)
                self.generation += 1
        except KeyboardInterrupt:
            pass

        # show final stats
        print('\nBest genome:\n{!s}'.format(self.population.fittest.genome))


if __name__ == '__main__':
    config = Config(num_inputs=5, num_outputs=3, pop_size=16, target_num_species=8)
    config.activation_config.options = "sin gauss identity sigmoid cube exp hat inv log softplus square tanh".split()
    print("Initializing simulation...")
    sim = Simulation(config)
    print("Running simulation...")
    sim.run()
