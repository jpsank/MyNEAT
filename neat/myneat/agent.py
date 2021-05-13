
from neat.base import BaseAgent
from neat.base.nn import RecurrentNetwork

from .world import *


class EntityAgent(BaseAgent, Entity):
    def __init__(self, genome):
        BaseAgent.__init__(self, genome)

        params = {k: v() for k, v in AGENT_SHAPE_INIT.items()}
        r, g, b = genome.body_genes[1], genome.body_genes[2], genome.body_genes[3]
        r, g, b = (int(gene.value*255) for gene in (r, g, b))
        Entity.__init__(self, shape=AGENT_SHAPE_TYPE(**params), color=(r, g, b))

        self.brain = RecurrentNetwork.create(genome)

        self.health = 0
        self.dead = False

    def draw(self, batch):
        yield super(EntityAgent, self).draw(batch)
        yield draw_line(*self.shape.pos, *(self.shape.pos + self.shape.dir * self.shape.radius * 2),
                        color=self.color, batch=batch)

    def move(self, forward, turn):
        self.shape.x += math.cos(self.shape.angle) * forward
        self.shape.y += math.sin(self.shape.angle) * forward
        self.shape.angle += turn

