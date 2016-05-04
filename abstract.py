import math
import numpy as np
import random
from constants import *

class ParentSprite(object):

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def get_center_x(self):
        """Gets the x coordinate of the center"""
        return self.center_x

    def get_center_y(self):
        """Gets the y coordinate of the center"""
        return self.center_y

    def get_dist(self, other):
        """
        Gets the distance between two abstract sprites
        """
        dist = np.hypot(
            other.get_center_x()-self.get_center_x(), other.get_center_y()-self.get_center_y())
        return dist

    def intersect(self, other):
        """
        tells whether or not two AbstractSprites are intersecting
        """
        dist = self.get_dist(other)
        return dist < self.radius + other.radius

    def angle_between(self, other):
        """
        Gets the angle between this sprite and another Abstract Sprite
        """
        deltaX = other.get_center_x() - self.get_center_x()
        deltaY = other.get_center_y() - self.get_center_y()
        return math.atan2(deltaY, deltaX)