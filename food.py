import random
from constants import *

class Food(object):
    """ Represents a piece of food in our game. """
    def __init__(self):
        """ Initializes a food object to a specified center and radius. """
        border = 20
        self.center_x = random.uniform(0 + border, SCREEN_SIZE[0] - border)
        self.center_y = random.uniform(0 + border, SCREEN_SIZE[1] - border)
        self.int_center = (int(self.center_x), int(self.center_y))
        self.radius = random.randint(5, 10)
        #don't use color right now.  Maybe in the future change blobs to
        #color of food that they are tracking
        # self.color = color
        self.eaten = False
