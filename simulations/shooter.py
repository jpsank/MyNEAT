
from neat.realtime.population import Agent, Population
from neat.base.nn.recurrent import RecurrentNetwork
from neat.realtime.config import Config

import pyglet
import random


WIDTH, HEIGHT = 700, 700
game_window = pyglet.window.Window(WIDTH, HEIGHT)


class Shooter(Agent):
    def __init__(self, genome):
        super().__init__(genome)
        self.brain = RecurrentNetwork.new(genome)
        self.x, self.y = random.random()*WIDTH, random.random()*HEIGHT


config = Config(num_inputs=3, num_outputs=3)
population = Population(config, Shooter)
population.init()


@game_window.event
def on_draw():
    game_window.clear()


def update(dt):
    for gid, agent in population.agents.items():
        inputs = [1, 2, 3]
        outputs = agent.brain.activate(inputs)
        agent.fitness += 0.1

    population.update()


if __name__ == '__main__':
    pyglet.clock.schedule_interval(update, 0.5)
    pyglet.app.run()




