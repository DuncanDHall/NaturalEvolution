import pygame, numpy, math
import random
from pygame.locals import QUIT, KEYDOWN
import time
from random import choice

class PyGameBrickView(object):
    """ Provides a view of the brick breaker model in a pygame
        window """
    def __init__(self, model, size):
        """ Initialize with the specified model """
        self.model = model
        self.screen = pygame.display.set_mode(size)


    def draw(self):
        """ Draw the game to the pygame window """
        # draw all the bricks to the screen
        self.screen.fill(pygame.Color('black'))

        for blob in self.model.blobs:
            pygame.draw.circle(
                self.screen, 
                pygame.Color('white'),
                (blob.center_x, blob.center_y),
                blob.radius
                )

        for food in self.model.foods:
            pygame.draw.circle(
                self.screen, 
                pygame.Color('orange'),
                (food.center_x, food.center_y),
                food.radius
                )

        pygame.display.update()


class Model(object):
    """ Represents the game state for brick breaker """
    def __init__(self, width, height):
        self.height = height
        self.width = width

        self.blobs = []
        self.foods = []
        #create blobs
        for i in range(0, 5):
            x = random.randint(0, 500)
            y = random.randint(0, 500)
            self.blobs.append(Blob(x, y, 10))
        #create foods
        for i in range(0, 10):
            x = random.randint(0, 500)
            y = random.randint(0, 500)
            radius = random.randint(5, 10)
            self.foods.append(Food(x, y, radius))
        
    def update(self):
        """ Update the model state """
        for blob in self.blobs:
            blob.update()



class Blob(object):
    """ Represents a ball in my brick breaker game """
    def __init__(self, center_x, center_y, radius):
        """ Create a ball object with the specified geometry """
        self.center_x = center_x
        self.center_y = center_y
        self.radius = radius
        self.velocity_x = 0         # pixels / frame
        self.velocity_y = -5         # pixels / frame
        self.energy = 10
        self.MAX_ENERGY = 50
        
        
    def intersect(self, other): 
        """
        Requires both objects to have center_x, center_y, and radius attributes
        """
        dist = abs(math.hypot(self.center_x-other.center_x, self.center_y-other.center_y))
        return dist < self.radius + other.radius

    def state_matrix(self):
        state = numpy.array([self.center_x, self.center_y, self.radius, self.velocity_x, self.velocity_y, self.energy])

    def update(self):
        """ Update the position of the ball due to time passing """
        self.center_x += self.velocity_x
        self.center_y += self.velocity_y


class Food(object):
    """ Represents a brick in my brick breaker game """
    def __init__(self, center_x, center_y, radius):
        """ Initializes a Brick object with the specified
            geometry and color """
        self.center_x = center_x
        self.center_y = center_y
        self.radius = radius
        #self.color = color


class PyGameKeyboardController(object):
    def __init__(self, model):
        self.model = model

    def handle_event(self, event):
        """ Look for left and right keypresses to
            modify the x position of the paddle """
        if event.type != KEYDOWN:
            return
        if event.key == pygame.K_LEFT:
            pass
        if event.key == pygame.K_RIGHT:
            pass
        if event.key == pygame.K_SPACE:
            global running
            running = False

if __name__ == '__main__':
    pygame.init()
    size = (500, 500)

    model = Model(size[0], size[1])
    view = PyGameBrickView(model, size)
    controller = PyGameKeyboardController(model)
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
            else:
                controller.handle_event(event)
        model.update()
        view.draw()
        time.sleep(.01)
