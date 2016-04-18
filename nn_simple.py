import pygame
import random
import time
from pygame.locals import QUIT, KEYDOWN
# from random import choice
# from math import pi, sin, cos, atan
from constants import *
from food import *
from blob import *
from constants import *


class PyGameView(object):
    """ Provides a view of the environment in a pygame
        window """
    def __init__(self, model, size):
        """ Initialize with the specified model """
        self.model = model
        self.screen = pygame.display.set_mode(size)

    def draw(self):
        """ Draw the simulation to the pygame window """
        # fill background
        self.screen.fill(pygame.Color('black'))

        # draw generation number
        basicfont = pygame.font.SysFont(None, 48)
        sim_num_string = basicfont.render(
            str(self.model.generation), True, (255, 255, 255))
        self.screen.blit(sim_num_string, (1, 1))

        # draw blobs
        for blob in self.model.blobs:
            if blob.alive:
                pygame.draw.circle(
                    self.screen,
                    pygame.Color('white'),
                    blob.int_center,
                    blob.radius
                    )

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

        # create foods
        for i in range(0, FOOD_NUM):
            x, y = (d/2 for d in SCREEN_SIZE)
            border = 20
            x = random.randint(0 + border, SCREEN_SIZE[0] - border)
            y = random.randint(0 + border, SCREEN_SIZE[1] - border)
            radius = random.randint(5, 10)
            self.foods.append(Food(x, y, radius))

        # create blobs
        for i in range(0, BLOB_NUM):
            x = random.randint(0, SCREEN_SIZE[0])
            y = random.randint(0, SCREEN_SIZE[1])
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
        """ Looks for keyboard events. """
        if event.type != KEYDOWN:
            return True
        if event.key == pygame.K_SPACE:
            return False
        if event.key == pygame.K_d:
            for tup in sorted(model.vip_genes)[-2:]:
                print tup
        if event.key == pygame.K_k:
            for blob in model.blobs:
                blob.energy = 0
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
                runnbing = False
            else:
                # handle event can end pygame loop
                if not controller.handle_event(event):
                    running = False
        model.update()
        if model.generation % SIM_SKIP_NUM == 0:
            view.draw()
            time.sleep(.001)

    # nn = NN()
    # z1 = np.array([-1, 1])
    # print nn.process(z1)
