import math
import numpy as np
import random
from constants import *

class ParentSprite(object):

    def __init__(self):
        """ 
        Initialize a random location for the sprite
        """
        self.center_x = random.randint(0, SCREEN_SIZE[0])
        self.center_y = random.randint(0, SCREEN_SIZE[1])

    def get_dist(self, other):
        """ 
        The distance between two abstract sprites

        Args:
            other (object) - the other sprite
        """
        dist = np.hypot(
            other.center_x-self.center_x, other.center_y-self.center_y)
        return dist

    def intersect(self, other):
        """
        Tells whether or not two AbstractSprites are intersecting
            
        Args: 
            other (object) - the other sprite
        """
        dist = self.get_dist(other)
        return dist < self.radius + other.radius

    def angle_between(self, other):
        """ 
        Gets the angle between this sprite and another Abstract Sprite
            
        Args:
            other (object): other the other sprite
        """
        deltaX = other.center_x - self.center_x
        deltaY = other.center_y - self.center_y
        return math.atan2(deltaY, deltaX)
