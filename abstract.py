import math
import numpy as np
import random
from constants import *

class ParentSprite(object):

    def __init__(self):
        """ Initialize a random location for the sprite
        """
        self.center_x = random.randint(0, SCREEN_SIZE[0])
        self.center_y = random.randint(0, SCREEN_SIZE[1])

    def get_center_x(self):
        """ Gets the x coordinate of the sprite
        """
        return self.center_x

    def get_center_y(self):
        """ Gets the y coordinate of the sprite
        """
        return self.center_y

    def get_dist(self, other):
        """ The distance between two abstract sprites
            Arugments: other - the other sprite
        """
        dist = np.hypot(
            other.get_center_x()-self.get_center_x(), other.get_center_y()-self.get_center_y())
        return dist

    def intersect(self, other):
        """ Tells whether or not two AbstractSprites are intersecting
            Arguments: other - the other sprite
        """
        dist = self.get_dist(other)
        return dist < self.radius + other.radius

    def angle_between(self, other):
        """ Gets the angle between this sprite and another Abstract Sprite
            Arguments: other the other sprite
        """
        deltaX = other.get_center_x() - self.get_center_x()
        deltaY = other.get_center_y() - self.get_center_y()
        return math.atan2(deltaY, deltaX)
