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
        super(Food, self).__init__(self.center_x, self.center_y)
        self.radius = random.randint(5, 10)
        #don't use color right now.  Maybe in the future change blobs to
        #color of food that they are tracking
        # self.color = color
        self.eaten = False