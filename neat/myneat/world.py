
from dataclasses import dataclass
from pyglet import shapes
from abc import ABC
import numpy as np
import math
import random


# COLORS
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)


def rect_rect_intersect(rect1, rect2):
    return rect1.x + rect1.width >= rect2.x and \
           rect1.x <= rect2.x + rect2.width and \
           rect1.y + rect1.height >= rect2.y and \
           rect1.y <= rect2.y + rect2.height


def circle_rect_intersect(circle, rect):
    test_x = circle.x
    test_y = circle.y
    rx, ry = rect.pos + rect.radius/2

    if circle.x < rx:
        test_x = rx
    elif circle.x > rx+rect.width:
        test_x = rx+rect.width
    if circle.y < ry:
        test_y = ry
    elif circle.y > ry + rect.height:
        test_y = ry + rect.height

    return distance(circle.x, circle.y, test_x, test_y) <= circle.radius


def circle_circle_intersect(circle1, circle2):
    return distance(*circle1.pos, *circle2.pos) <= circle1.radius + circle2.radius


def distance(x1, y1, x2, y2):
    return math.sqrt((x1-x2)**2 + (y1-y2)**2)


@dataclass
class Shape(ABC):
    x: float
    y: float
    angle: float

    @property
    def radius(self): return

    @property
    def dir(self):
        return np.array([math.cos(self.angle), math.sin(self.angle)])

    @property
    def pos(self):
        return np.array([self.x, self.y])

    def draw(self, batch, color=WHITE): pass
    def intersect(self, other): pass


@dataclass
class Rect(Shape):
    width: float
    height: float

    @property
    def radius(self):
        return np.array([self.width, self.height])

    def draw(self, batch, color=WHITE):
        rectangle = shapes.Rectangle(self.x, self.y, self.width, self.height, color=color, batch=batch)
        rectangle.rotation = self.angle
        return rectangle

    def intersect(self, other):
        if isinstance(other, Circle):
            return circle_rect_intersect(other, self)
        elif isinstance(other, Rect):
            return rect_rect_intersect(self, other)


@dataclass
class Circle(Shape):
    radius: float

    def draw(self, batch, color=WHITE):
        return shapes.Circle(self.x, self.y, self.radius, color=color, batch=batch)

    def intersect(self, other):
        if isinstance(other, Circle):
            return circle_circle_intersect(self, other)
        elif isinstance(other, Rect):
            return circle_rect_intersect(self, other)


class Entity(ABC):
    def __init__(self, shape: Shape, color):
        self.shape = shape
        self.color = color

    def draw(self, batch):
        return self.shape.draw(batch, color=self.color)


def draw_line(x, y, x2, y2, width=1, color=WHITE, batch=None):
    return shapes.Line(x, y, x2, y2, width, color=color, batch=batch)


class Box(Entity):
    def __init__(self):
        params = {k: v() for k, v in BOX_SHAPE_INIT.items()}
        super().__init__(shape=BOX_SHAPE_TYPE(**params), color=BOX_COLOR_INIT())


# GAME
WIDTH, HEIGHT = 700, 700

# AGENTS
AGENT_SHAPE_TYPE = Circle
AGENT_SHAPE_INIT = {
    "x": lambda: random.random()*WIDTH,
    "y": lambda: random.random()*HEIGHT,
    "angle": lambda: random.random()*2*math.pi,
    "radius": lambda: 10
}

# BOXES
BOX_SHAPE_TYPE = Rect
BOX_SHAPE_INIT = {
    "x": lambda: random.random()*WIDTH,
    "y": lambda: random.random()*HEIGHT,
    "angle": lambda: random.random()*2*math.pi,
    "width": lambda: 10,
    "height": lambda: 10
}
BOX_COLOR_INIT = lambda: BLUE

