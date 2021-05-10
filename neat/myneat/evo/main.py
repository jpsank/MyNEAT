import cProfile

from neat.myneat.population import Population
from neat.myneat.config import Config
from neat.base.data import Index

from neat.myneat.world import *

import pyglet
import random
import math


config = Config(num_inputs=4, num_outputs=3, pop_size=100, target_num_species=20, max_stagnation=200)
population = Population(config)

# Initialize population
population.init()

# Create boxes
boxes = []
for _ in range(20):
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
        closest_i, closest, closest_dist = None, None, None
        # for agent2 in population.agents.values():
        #     dist = distance(*agent.shape.pos, *agent2.shape.pos)
        #     if closest_dist is None or dist < closest_dist:
        #         closest, closest_dist = agent2, dist
        for i, box in enumerate(boxes):
            dist = distance(*agent.shape.pos, *box.shape.pos)
            if closest_dist is None or dist < closest_dist:
                closest_i, closest, closest_dist = i, box, dist

        # angle = angle_between(*agent.shape.pos, *closest.shape.pos)
        # angle = angle_diff(agent.shape.angle, angle)
        # inputs = [*closest.color,
        #           1 / (closest_dist / agent.shape.radius),
        #           angle,
        #           1]
        inputs = [(agent.shape.x - closest.shape.x)/agent.shape.radius,
                  (agent.shape.y - closest.shape.y)/agent.shape.radius,
                  agent.shape.angle,
                  1]

        outputs = agent.brain.activate(inputs)
        forward, turn, _ = outputs
        agent.move(forward * 2, turn/2)

        if agent.shape.intersect(closest.shape):
            agent.fitness += 1
            del boxes[closest_i]
            boxes.append(Box())

    population.update()


def run():
    pyglet.clock.schedule_interval(update, 0.01)
    pyglet.app.run()


if __name__ == '__main__':
    cProfile.run("run()", sort='tottime')

