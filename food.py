import random
from constants import *
from abstract import ParentSprite

class Food(ParentSprite):
    """ Represents a piece of food in our game. """
    def __init__(self):
        """ Initializes a food object to a specified center and radius. """
        border = 20
        self.center_x = random.randint(0 + border, SCREEN_SIZE[0] - border)
        self.center_y = random.randint(0 + border, SCREEN_SIZE[1] - border)
        self.radius = random.randint(5, 10)
        #don't use color right now.  Maybe in the future change blobs to
        #color of food that they are tracking
        # self.color = color
        self.eaten = False
    def get_center_x(self):
        """Gets the x coordinate of the center"""
        return self.center_x
        
    def get_center_y(self):
        """Gets the y coordinate of the center"""
        return self.center_y
