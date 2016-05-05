import random
from constants import *
from abstract import ParentSprite



class Food(ParentSprite):
    """ 
    Represents a piece of food in our game. Inherits from ParentSprite
    """


    def __init__(self):
        """ 
        Initializes a food object to a specified center and radius. 
        """
        super(Food, self).__init__()
        self.radius = random.randint(5, 10)
        self.eaten = False


