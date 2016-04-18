
class Food(object):
    """ Represents a piece of food in our game. """
    def __init__(self, center_x, center_y, radius):
        """ Initializes a food object to a specified center and radius. """
        self.center_x = center_x
        self.center_y = center_y
        self.radius = radius
        # self.color = color
        self.eaten = False
