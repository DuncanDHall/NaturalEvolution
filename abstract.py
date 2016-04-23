import math
import random
from constants import *

class ParentSprite(object):

    def get_center_x(self):
        """Gets the x coordinate of the center"""
        #return a random value within constants screen size
        return random.randint(0, SCREEN_SIZE[0]) # or is it 1????

    def get_center_y(self):
        """Gets the y coordinate of the center"""
        #return a random value within constants screen size
        return random.randint(0, SCREEN_SIZE[1]) #or is is 0..see above method

    def get_dist(self, other):
        """ 
        Gets the distance between two abstract sprites
        """
        dist = abs(math.hypot(
            self.get_center_x()-other.get_center_x(), self.get_center_y()-other.get_center_y()))
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