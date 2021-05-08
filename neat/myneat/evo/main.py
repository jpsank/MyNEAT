import cProfile

from neat.myneat.population import Population
from neat.myneat.config import Config
from neat.base.data import Index

from neat.myneat.world import *

import pyglet
import random
import math


config = Config(num_inputs=5, num_outputs=3, pop_size=100, target_num_species=40)
population = Population(config)

# Initialize population
population.init()


# Create boxes
boxes = []
for _ in range(4):
    boxes.append(Box())


window = pyglet.window.Window(WIDTH, HEIGHT)


@window.event
def on_draw():
    batch = pyglet.graphics.Batch()
    things = []
    for agent in population.agents.values():
        things.extend(agent.draw(batch))

    for box in boxes:
        things.append(box.draw(batch))

    window.clear()
    batch.draw()


def update(dt):
    for agent in population.agents.values():
        inputs = [1, 2, 3, 4, 5]

        outputs = agent.brain.activate(inputs)
        forward, turn, _ = outputs
        agent.move(agent, forward*10, turn)

    population.update()


def run():
    pyglet.clock.schedule_interval(update, 0.01)
    pyglet.app.run()


if __name__ == '__main__':
    cProfile.run("run()", sort='tottime')

