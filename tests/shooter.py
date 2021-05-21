
from neat.realtime import Config, BaseAgent, BaseGenome, Population
from neat.base.nn import FeedForwardNetwork

from shapely.geometry import LineString
from shapely.geometry import Point

from itertools import count
from collections import defaultdict
import pyglet
from pyglet import shapes
import random
import numpy as np
import math


RED = (255, 0, 0)
BLUE = (0, 0, 255)


def distance(x1, y1, x2, y2):
    return math.sqrt((x1-x2)**2 + (y1-y2)**2)


# https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection
def line_intersection(x1, y1, x2, y2, x3, y3, x4, y4, line1='segment', line2='segment'):
    # denominator
    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if den == 0:
        return None

    # formulas
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
    u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / den

    cond1 = (line1 == 'segment' and 0 <= t <= 1) or (line1 == 'ray' and 0 <= t) or line1 == 'line'
    cond2 = (line2 == 'segment' and 0 <= u <= 1) or (line2 == 'ray' and 0 <= u) or line2 == 'line'
    if cond1 and cond2:
        px = x1 + t * (x2 - x1)
        py = y1 + t * (y2 - y1)
        return np.array([px, py])


def segment_circle_intersection(x1, y1, x2, y2, cx, cy, radius):
    p = Point(cx, cy)
    c = p.buffer(radius).boundary
    l = LineString([(x1, y1), (x2, y2)])
    i = c.intersection(l)
    if i.is_empty:
        return None
    if isinstance(i, Point):
        return i.coords[0]
    else:
        return i[0].coords[0]


class Entity:
    def __init__(self, x, y, angle):
        self.x, self.y = x, y
        self.angle = angle

        self.type = ''

    @property
    def dir(self):
        return np.array([math.cos(self.angle), math.sin(self.angle)])

    @property
    def pos(self):
        return np.array([self.x, self.y])


class Ray(Entity):
    def __init__(self, x, y, angle):
        super().__init__(x, y, angle)

        self.type = 'ray'

    def cast(self, other):
        x1, y1 = self.pos
        x2, y2 = self.pos + self.dir
        if other.type == 'ray':
            x3, y3 = other.pos
            x4, y4 = other.pos + other.dir
            return line_intersection(x1, y1, x2, y2, x3, y3, x4, y4, 'ray', 'ray')
        elif other.type == 'rect':
            corners = np.array([(-1, -1), (-1, 1), (1, 1), (1, -1)])
            closest_pt = None
            closest_dist = None
            for i in range(4):
                x3, y3 = other.pos + corners[i-1] * other.radius
                x4, y4 = other.pos + corners[i] * other.radius
                pt = line_intersection(x1, y1, x2, y2, x3, y3, x4, y4, 'ray', 'segment')
                if pt is not None:
                    dist = np.linalg.norm(pt - self.pos)
                    if closest_dist is None or dist < closest_dist:
                        closest_pt, closest_dist = pt, dist
            return closest_pt
        elif other.type == 'circle':
            return segment_circle_intersection(x1, y1, *(self.pos + self.dir * 1000),
                                               other.x, other.y, other.radius)


WIDTH, HEIGHT = 640, 640
window = pyglet.window.Window(WIDTH, HEIGHT)


class Shooter(BaseAgent, Entity):
    def __init__(self, genome):
        BaseAgent.__init__(self, genome)
        self.brain = FeedForwardNetwork.create(genome)
        Entity.__init__(self, random.random()*WIDTH, random.random()*HEIGHT, 0)

        self.vx, self.vy = 0, 0

        self.reload = 0
        self.dead = False

        self.color = RED
        self.team = None
        self.type = 'circle'
        self.radius = 5

    def loaded(self):
        return self.reload <= 0

    def shoot(self):
        self.reload = 20
        return Ray(self.x, self.y, self.angle)

    def move(self, forward, a):
        self.vx = math.cos(self.angle) * forward
        self.vy = math.sin(self.angle) * forward
        self.angle += a

    def update(self):
        if self.reload > 0:
            self.reload -= 1

        self.x += self.vx
        self.y += self.vy

    def draw(self, x, y):
        yield shapes.Circle(x + self.x, y + self.y, self.radius, color=self.color, batch=batch)
        yield shapes.Line(x + self.x, y + self.y,
                          x + self.x + math.cos(self.angle) * self.radius * 2,
                          y + self.y + math.sin(self.angle) * self.radius * 2,
                          self.radius, color=self.color, batch=batch)


class Game:
    def __init__(self, width, height):
        self.width, self.height = width, height
        self.agents = []
        self.done = False
        self.tid_indexer = count()

        self.draw_data = []

    @staticmethod
    def random_color():
        return int(random.random()*255), int(random.random()*255), int(random.random()*255)

    def random_pos(self):
        return random.random() * self.width, random.random() * self.height

    def add_team(self, members):
        tid = next(self.tid_indexer)
        color = Game.random_color()
        cx, cy = self.random_pos()
        for agent in members:
            agent.team = tid
            agent.color = color
            agent.x, agent.y = cx + random.random() * self.width / 4, cy + random.random() * self.height / 4
            self.agents.append(agent)

    def init(self, teams):
        for members in teams:
            self.add_team(members)

    def add_agent(self, agent):
        if self.agents:
            teams = defaultdict(list)
            for agent in self.agents:
                teams[agent.team].append(agent)

            if len(teams) > 1:
                least_populated_id = min(teams.keys(), key=lambda k: len(teams[k]))
                agent.team = least_populated_id

                fellow = teams[least_populated_id][0]
                agent.color = fellow.color
                agent.x = fellow.x + random.random() * self.width / 4
                agent.y = fellow.y + random.random() * self.height / 4
                self.agents.append(agent)
                return

        self.add_team([agent])

    @staticmethod
    def draw_line(x1, y1, x2, y2, width, color, x=0, y=0):
        return shapes.Line(x+x1, y+y1, x+x2, y+y2, width, color=color, batch=batch)

    @staticmethod
    def draw_circle(x1, y1, radius, color, x=0, y=0):
        return shapes.Circle(x + x1, y + y1, radius, color=color, batch=batch)

    def do_shoot(self, agent, enemies):
        draw_data = []
        ray = agent.shoot()
        closest, closest_dist, closest_pt = None, None, None
        for enemy in enemies:
            pt = ray.cast(enemy)
            if pt is not None:
                dist = np.linalg.norm(pt - agent.pos)
                if closest is None or dist < closest_dist:
                    closest_pt, closest_dist, closest = pt, dist, enemy
        if closest is None:
            draw_data.append([Game.draw_circle,
                              *(ray.pos + ray.dir * agent.radius * 2), 4, agent.color])
            agent.fitness -= .1
        else:
            draw_data.append([Game.draw_line,
                              *ray.pos, *closest.pos, 5, agent.color])
            draw_data.append([Game.draw_circle,
                              closest.x, closest.y, 8, agent.color])
            agent.fitness += 1
            closest.fitness -= 1
            # closest.dead = True
        return draw_data

    def step(self):
        alive = [agent for agent in self.agents if not agent.dead]
        if len(alive) == 0:
            self.done = True
            return
        for agent in alive:
            forward, turn, shoot = 0, 0, 0
            enemies = (enemy for enemy in alive if enemy.team != agent.team)
            for enemy in enemies:
                dx, dy = agent.x - enemy.x, agent.y - enemy.y
                dx, dy = 1 if abs(dx) < 1 else 1 / dx, 1 if abs(dy) < 1 else 1 / dy
                inputs = [dx, dy, agent.angle, 1]

                outputs = agent.brain.activate(inputs)
                if outputs[0] > 0.5:
                    forward += outputs[1]
                    turn += outputs[2]
                    shoot += outputs[3]

            if shoot > 0.9 and agent.loaded():
                self.draw_data.extend(self.do_shoot(agent, enemies))

            agent.move(forward, turn)
            agent.update()

            r = agent.radius
            if (agent.x < r and agent.vx < 0) or (agent.x > self.width-r and agent.vx > 0):
                agent.vx = -agent.vx
            if (agent.y < r and agent.vy < 0) or (agent.y > self.height-r and agent.vy > 0):
                agent.vy = -agent.vx
            agent.x = r if agent.x < r else self.width-r if agent.x > self.width-r else agent.x
            agent.y = r if agent.y < r else self.height-r if agent.y > self.height-r else agent.y

    def draw(self, x, y):
        yield shapes.BorderedRectangle(x, y, self.width, self.height, color=(0, 0, 0),
                                       border=1, border_color=(255, 255, 0), batch=batch)

        for data in self.draw_data:
            func, data = data[0], data[1:]
            yield func(*data, x=x, y=y)
        self.draw_data.clear()

        for agent in (a for a in self.agents if not a.dead):
            yield from agent.draw(x, y)


config = Config(num_inputs=4, num_outputs=4, max_stagnation=500, pop_size=128, target_num_species=20)
population = Population(config, agent_type=Shooter)
population.init()


batch = pyglet.graphics.Batch()
to_draw = []

GAME_WIDTH, GAME_HEIGHT = 160, 160
games = []


@window.event
def on_draw():
    window.clear()

    x, y = 0, 0
    for i, game in enumerate(games):
        to_draw.extend(game.draw(x, y))
        x += GAME_WIDTH
        if x + GAME_WIDTH > WIDTH:
            x = 0
            y += GAME_HEIGHT
    batch.draw()

    to_draw.clear()


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def update(dt):
    for game in games:
        game.step()

    population.update()
    # print("Best fitness:", population.fittest.fitness)

    agents = population.agents.values()
    for game in games:
        game.agents = [a for a in game.agents if a in agents]
    for agent in agents:
        if agent.team is None:
            game_to_join = min(games, key=lambda g: len(g.agents))
            game_to_join.add_agent(agent)


def init():
    agents = list(population.agents.values())
    random.shuffle(agents)
    teams = list(chunks(agents, 4))
    for team_pair in chunks(teams, 2):
        game = Game(GAME_WIDTH, GAME_HEIGHT)
        game.init(team_pair)
        games.append(game)


if __name__ == '__main__':
    init()
    pyglet.clock.schedule_interval(update, 0.01)
    pyglet.app.run()




