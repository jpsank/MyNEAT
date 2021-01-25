""" Demonstrate multiple interacting populations with distinct genome configurations in one simulation (WIP). """

from myneat.myneat.population import Agent, Population
from myneat.myneat.config import Config


class Wolf(Agent):
    def __init__(self, x, y, genome):
        super().__init__(genome)
        self.x, self.y = x, y


class Elk(Agent):
    def __init__(self, x, y, genome):
        super().__init__(genome)
        self.x, self.y = x, y


class Grass(Agent):
    def __init__(self, x, y, genome):
        super().__init__(genome)
        self.x, self.y = x, y


wolf_config = Config()
elk_config = Config()
grass_config = Config()

wolf_population = Population(wolf_config, Wolf)
elk_population = Population(elk_config, Elk)
grass_population = Population(grass_config, Grass)


