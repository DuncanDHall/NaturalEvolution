from constants import *
from food import *
import random
from nn import NN
import numpy as np
import math


class Blob(object):
    """ Represents a blob. """
    def __init__(self, target, nn=None):
        """ Create a ball object with the specified geometry """
        self.center_x = random.randint(0, SCREEN_SIZE[0])
        self.center_y = random.randint(0, SCREEN_SIZE[1])
        self.int_center = int(self.center_x), int(self.center_y)
        self.radius = random.randint(5, 10)
        self.velocity_x = 0         # pixels / frame
        self.velocity_y = 0         # pixels / frame
        self.MAX_VELOCITY = 5
        self.energy = 100
        self.MAX_ENERGY = 100
        self.alive = True
        self.food_eaten = 0
        self.score_int = 0
        self.target = target

        # direction changes:
        # self.direction = random.uniform(0, 2*pi)

        # Neural Network stuff here:
        if nn is not None:
            self.nn = nn
        else:
            self.nn = NN()

    def intersect(self, other):
        """ Returns true if two objects intersect.Requires both objects to
            have center_x, center_y, and radius attributes
        """
        dist = abs(math.hypot(
            self.center_x-other.center_x, self.center_y-other.center_y))
        return dist < self.radius + other.radius

    def update(self, model):
        """ Update the position of a blob and evaluates its energy. """
        self.center_x += self.velocity_x
        self.center_y += self.velocity_y
        # self.center_x += self.energy/50 * cos(self.direction)
        # self.center_y += self.energy/50 * sin(self.direction)

        self.int_center = int(self.center_x), int(self.center_y)

        self.energy -= .1
        if self.energy < 0:
            self.alive = False
            self.score_int = self.score()
            model.vip_genes.append((self.score_int, self.nn))

            model.blobs.remove(self)

        self.change_vel()

        for i in range(len(model.foods)-1, -1, -1):
            f = model.foods[i]
            if self.intersect(f):
                self.food_eaten += 1
                self.energy += 0
                if self.energy > self.MAX_ENERGY:
                    self.energy = self.MAX_ENERGY

                del model.foods[i]
                model.foods.append(
                    Food(
                        random.randint(10, SCREEN_SIZE[0]-10),
                        random.randint(10, SCREEN_SIZE[1]-10),
                        random.randint(5, 10)))

        self.target = model.foods[0]

    def change_vel(self):
        env = np.array([
            self.center_x - self.target.center_x,
            self.center_y - self.target.center_y])
        acceleration_x, acceleration_y = tuple(self.nn.process(env))

        self.velocity_x = acceleration_x
        self.velocity_y = acceleration_y

        # self.direction = atan(acceleration_x/acceleration_y)

        # self.direction = atan(acceleration_x/acceleration_y)

        if abs(self.velocity_x) > self.MAX_VELOCITY:
            self.velocity_x = (
                self.velocity_x/abs(self.velocity_x)
                )*self.MAX_VELOCITY
        if abs(self.velocity_y) > self.MAX_VELOCITY:
            self.velocity_y = (
                self.velocity_y/abs(self.velocity_y)
                )*self.MAX_VELOCITY

    def score(self):
        return self.food_eaten
