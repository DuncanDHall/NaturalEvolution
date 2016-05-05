import pygame
import random
import time
from pygame.locals import QUIT, KEYDOWN
from constants import *
from food import *
from blob import *
from constants import *
import os


class PyGameView(object):
    """ 
    Provides a view of the environment in a pygame window 
    """


    def __init__(self, model, size):
        """ 
        Initialize model
        """
        self.model = model
        self.screen = pygame.display.set_mode(size)


    def draw_text(self, text, x, y, size, color=(100, 100, 100)):
        """ 
        Helper to draw text onto screen.

        Args:
            text (string): text to display
            x (int): horizontal position
            y (int): vertical position
            size (int): font size
            color (3-tuple int): color of text.  Can use pygame colors.
            defualt = (100, 100, 100)
        """
        basicfont = pygame.font.SysFont(None, size)
        text_render = basicfont.render(
            text, True, color)
        self.screen.blit(text_render, (x, y))


    def draw(self):
        """ 
        Draw blobs, food, and text to the pygame window 
        """
        # fill background
        self.screen.fill(pygame.Color('black'))

        # draw population number
        self.draw_text(str(self.model.population), 1, 1, 48)

        # draw controls helper
        if model.show_controls:
            for n, line in enumerate(CONTROLS):
                self.draw_text(line, 10, 50+14*n, 20)
        else:
            self.draw_text(
                "h = toggle help", 30, 1, 20)

        # draw blobs
        for blob in self.model.blobs:
            if blob.alive:
                pygame.draw.circle(
                    self.screen,
                    pygame.Color(blob.color, blob.color, blob.color),
                    blob.int_center,
                    blob.radius
                    )
                pygame.draw.line(
                    self.screen,
                    pygame.Color('red'),
                    blob.int_center,
                    (int(blob.center_x + 20*np.cos(blob.angle)), int(blob.center_y) + 20*np.sin(blob.angle)),
                    1)
                if model.draw_sight: # sight lines are toggleable
                    pygame.draw.line(
                        self.screen,
                        pygame.Color('green'),
                        blob.int_center,
                        (int(blob.center_x + blob.sight_radius*np.cos(blob.angle-blob.sight_angle)), int(blob.center_y) + blob.sight_radius*np.sin(blob.angle - blob.sight_angle)),
                        1)
                    pygame.draw.line(
                        self.screen,
                        pygame.Color('green'),
                        blob.int_center,
                        (int(blob.center_x + blob.sight_radius*np.cos(blob.angle+blob.sight_angle)), int(blob.center_y) + blob.sight_radius*np.sin(blob.angle + blob.sight_angle)),
                        1)
        # draw food
        for food in self.model.foods:
            pygame.draw.circle(
                self.screen,
                pygame.Color('orange'),
                (food.center_x, food.center_y),
                food.radius
                )

        pygame.display.update()



class Model(object):
    """ 
    Represents the state of all entities in the environment and drawing
    parameters
    """


    def __init__(self, width, height):
        """
        initialize model, environment, and default keyboard controller states

        Args:
            width (int): width of window in pixels
            height (int): height of window in pixels
        """
        #window parameters / drawing
        self.height = height
        self.width = width
        self.show_gen = True #show generation number
        self.show_controls = False #controls toggle
        self.draw_sight = False #draw sight lines
        self.sleep_time = .005 #seconds between frames

        # create foods
        self.foods = []
        for i in range(0, FOOD_NUM):
            self.foods.append(Food())

        # create blobs
        self.blobs = []
        for i in range(0, BLOB_NUM):
            self.blobs.append(Blob())

        # population progressions
        self.population = 0
        self.vip_genes = []


    def update(self):
        """ 
        Update the model state. Each time update is called can be though of as
        a frame 
        """
        for blob in reversed(self.blobs):
            blob.update(self)

        # If all blobs are dead, start new cycle
        if self.blobs == []:
            self.create_population(NUM_PARENTS)

            self.vip_genes = []
            self.population += 1
            if self.population % 10 == 0:
                print 'population {} complete'.format(self.population)


    def create_population(self, num_winners=2):
        """ 
        create new population of blobs based on the top scoring blobs

        Args:
            num_winners (int): number of vip blobs to use
        """
        top_scoring = sorted(self.vip_genes, reverse=True)[:num_winners]

        for i in range(0, BLOB_NUM):
            new_NN = NN(parents_NN=top_scoring)
            self.blobs.append(Blob(new_NN))



class PyGameKeyboardController(object):
    """
    Keyboard controller that responds to keyboard input
    """


    def __init__(self, model):
        """
        Creates keyboard controller

        Args:
            model (object): contains attributes of the environment
        """
        self.model = model


    def handle_event(self, event):
        """ 
        Look for left and right keypresses to modify the x position of the paddle 

        Args:
            event (pygame class): type of event
        """
        if event.type != KEYDOWN:
            return True
        elif event.key == pygame.K_SPACE:
            return False
        elif event.key == pygame.K_d:
            for blob in model.blobs:
                print 'W1 is'
                print blob.nn.W1
                print ""
                print "W2 is"
                print ""
                print blob.nn.W2
                # break #iterate through first thing in a list
        elif event.key == pygame.K_k:
            for blob in model.blobs:
                blob.energy = 0
        elif event.key == pygame.K_s:
            model.show_gen = not model.show_gen
        elif event.key == pygame.K_PERIOD:
            model.sleep_time = max(model.sleep_time-0.005, 0.0)
        elif event.key == pygame.K_COMMA:
            model.sleep_time += 0.005
        elif event.key == pygame.K_h:
            model.show_controls = not model.show_controls
        elif event.key == pygame.K_a:
            model.draw_sight = not model.draw_sight
        return True



if __name__ == '__main__':
    pygame.init()
    size = SCREEN_SIZE

    model = Model(size[0], size[1])
    view = PyGameView(model, size)
    controller = PyGameKeyboardController(model)
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
            else:
                # handle event can end pygame loop
                if not controller.handle_event(event):
                    running = False
        model.update()
        if model.show_gen:
            view.draw()
            time.sleep(model.sleep_time)