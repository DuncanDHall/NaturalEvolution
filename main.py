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
    """ Provides a view of the environment in a pygame
        window """
    def __init__(self, model, size):
        """ Initialize with the specified model """
        self.model = model
        self.screen = pygame.display.set_mode(size)

    def draw_text(self, text, x, y, size, color=(100, 100, 100)):
        """ helper to draw text (string input) onto screen at coords (x, y)
            and specified font size and color
        """
        basicfont = pygame.font.SysFont(None, size)
        text_render = basicfont.render(
            text, True, color)
        self.screen.blit(text_render, (x, y))

    def draw(self):
        """ Draw the simulation to the pygame window """
        # fill background
        self.screen.fill(pygame.Color('black'))

        # draw generation number
        self.draw_text(str(self.model.generation), 1, 1, 48)

        # draw controls helper
        if model.show_key:
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
    """ Represents the state of all entities in the environment"""
    def __init__(self, width, height):
        self.height = height
        self.width = width

        self.blobs = []
        self.foods = []
        self.vip_genes = []
        self.generation = 0

        self.show_gen = True
        self.show_key = False

        # create foods
        for i in range(0, FOOD_NUM):
            self.foods.append(Food())

        # create blobs
        for i in range(0, BLOB_NUM):
            self.blobs.append(Blob(self.foods[0]))

    def update(self):
        """ Update the model state """
        for blob in reversed(self.blobs):
            blob.update(self)

        # If all blobs are dead, start new cycle
        if self.blobs == []:
            self.create_generation(NUM_PARENTS)

            self.vip_genes = []
            self.generation += 1
            if self.generation % 10 == 0:
                print 'generation {} complete'.format(self.generation)

    def create_generation(self, num_winners=2):
        """ Handles gene mutation and recombination"""
        top_scoring = sorted(self.vip_genes, reverse=True)[:num_winners]

        for i in range(0, BLOB_NUM):
            new_NN = NN(parents_NN=top_scoring)
            # TODO: Check if dna results are the blobs or others >> hmm?
            self.blobs.append(Blob(self.foods[0], new_NN))


class PyGameKeyboardController(object):
    def __init__(self, model):
        self.model = model

    def handle_event(self, event):
        """ Look for left and right keypresses to
            modify the x position of the paddle """
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
            global sleep
            sleep = max(sleep-0.02, 0.0)
        elif event.key == pygame.K_COMMA:
            global sleep
            sleep += 0.02
        elif event.key == pygame.K_h:
            model.show_key = not model.show_key

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
            time.sleep(sleep)

    # nn = NN()
    # z1 = np.array([-1, 1])
    # print nn.process(z1)
