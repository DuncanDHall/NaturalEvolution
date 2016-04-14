import pygame
import math
import random
import time
from pygame.locals import QUIT, KEYDOWN
from random import choice
import numpy as np

SCREEN_SIZE = (500, 500)
FOOD_NUM = 1
BLOB_NUM = 10
SIM_SKIP_NUM = 100  # the number of simulations you want to skip
NUM_PARENTS = 2

NUM_NODES = 2


class PyGameView(object):
    """ Provides a view of the environment in a pygame
        window """
    def __init__(self, model, size):
        """ Initialize with the specified model """
        self.model = model
        self.screen = pygame.display.set_mode(size)

    def draw(self):
        """ Draw the game to the pygame window """
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

        # multiplies positions vector by DNA to produce velocity
        # changes self.vele[1])

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

        # take a random gene from one of the parents
        # for i in range(average_dna.shape[0]):
        #     for j in range( average_dna.shape[1]):
        #         average_dna[i][j]=random.choice([top_scoring[0][0][i][j],
        #             top_scoring[1][0][i][j]])

        # #mutate one, make a new blob
        # for i in range(0, BLOB_NUM):
        #     x = random.randint(0, SCREEN_SIZE[0])
        #     y = random.randint(0, SCREEN_SIZE[1])
        #     mutated_dna = np.copy(average_dna)
        #     if random.random()<.6: #mutation chance for altering a gene
        #         mutation = (random.random()-.5)*2*(10**-7)
        #         mutated_dna[
        #             random.randrange(average_dna.shape[0])
        #         ][
        #             random.randrange(average_dna.shape[1])
        #         ] += mutation
        #     elif random.random()<0.4: #mutation chance for replacing a gene
        #         mutation = (random.random()-.5)*2*(10**-5)
        #         mutated_dna[
        #               random.randrange(average_dna.shape[0])
        #           ][
        #               random.randrange(average_dna.shape[1])
        #           ] += mutation
        #     self.blobs.append(Blob(x, y, 10, mutated_dna, self.foods[0]))


class NN(object):
    """ Represents the Neural Network of a blob """
    def __init__(self, parents_NN=None):
        """ this neural network takes in difference in x and y position between
            the agent and a single food entity.
            parents_NN should be passed in as a tuple of NN objects
        """

        self.inputLayerSize = 1
        self.outputLayerSize = 3
        self.hiddenLayerSize = 4
        self.W1 = np.random.uniform(-1, 1, (self.inputLayerSize, self.hiddenLayerSize))
        self.W2 = np.random.uniform(-1, 1, (self.hiddenLayerSize, self.outputLayerSize))

    def process(self, z1):
        """ propigates the signal through the neural network """
        # input and output to level 2 (nodes)
        z2 = z1.dot(self.W1)
        a2 = self.sigmoid(z2)
        # input and output to level 3 (results)
        z3 = a2.dot(self.W2)
        a3 = self.sigmoid(z3)

        return np.argmax(a3)

    def sigmoid(self, z):
        # Apply sigmoid activation function (arctan):
        # TODO: why does sigmoid not work?
        return 1/(1+np.exp(-z))


class Blob(object):
    """ Represents a ball in my natural evolution simulation """
    def __init__(self, target, nn=None):
        """ Create a ball object with the specified geometry """
        self.center_x = random.randint(0, SCREEN_SIZE[0])
        self.center_y = random.randint(0, SCREEN_SIZE[1])
        self.int_center = int(self.center_x), int(self.center_y)
        self.radius = random.randint(5, 10)
        self.angle = 30  * (np.pi / 180) #random.uniform(0,np.pi)
        self.MAX_VELOCITY = 5
        self.energy = 100
        self.MAX_ENERGY = 100
        self.alive = True
        self.food_eaten = 0
        self.score_int = 0
        self.target = target

        # Neural Network stuff here:
        if nn is not None:
            self.nn = nn
        else:
            self.nn = NN()

    def intersect(self, other):
        """ Requires both objects to have center_x, center_y, and radius
            attributes
        """
        dist = abs(math.hypot(
            self.center_x-other.center_x, self.center_y-other.center_y))
        return dist < self.radius + other.radius

    def update(self, model):
        """ Update the position of the blob due to time passing """
        deltaX = self.target.center_x - self.center_x
        deltaY = self.target.center_y - self.center_y
        env = np.array([
            self.angle - np.arctan(deltaX/deltaY)
            ])
        decision = self.nn.process(env)

        if decision == 0: #move forward
            self.center_x += self.MAX_VELOCITY * np.cos(self.angle)
            self.center_y += self.MAX_VELOCITY * np.sin(self.angle)
            self.int_center = int(self.center_x), int(self.center_y)
        if decision == 1: #turn counter clockwise
            self.angle -= .05
        if decision == 2: #turn clockwise
            self.angle += .05

        if self.angle > 2*np.pi:
            self.angle = self.angle % np.pi       
        if self.angle < -2*np.pi:
            self.angle = -self.angle % np.pi 

        self.energy -= 0
        if self.energy < 0:
            self.alive = False
            self.score_int = self.score()
            model.vip_genes.append((self.score_int, self.nn))

            model.blobs.remove(self)

        # self.change_vel()

        for i in range(len(model.foods)-1, -1, -1):
            f = model.foods[i]
            if self.intersect(f):
                self.food_eaten += 1
                self.energy += 0
                if self.energy > self.MAX_ENERGY:
                    self.energy = self.MAX_ENERGY

                del model.foods[i]
                # global SCREEN_SIZE
                model.foods.append(
                    Food(
                        random.randint(10, SCREEN_SIZE[0]-10),
                        random.randint(10, SCREEN_SIZE[1]-10),
                        random.randint(5, 10)))

        self.target = model.foods[0]

    # def change_vel(self): #change to turn left, turn right, or move forward
        # env = np.array([
        #     self.center_x - self.target.center_x,
        #     self.center_y - self.target.center_y])
    #     acceleration_x, acceleration_y = tuple(self.nn.process(env))

    #     # positions = np.array([
    #     #     self.center_x, self.center_y,
    #     # change matrix dimensions in model init and update:
    #     #     # self.velocity_x, self.velocity_y,
    #     #     self.target.center_x, self.target.center_y])
    #     # acceleration_x, acceleration_y = tuple(self.DNA.dot(positions))

    #     self.velocity_x = acceleration_x/10
    #     self.velocity_y = acceleration_y/10

    #     if abs(self.velocity_x) > self.MAX_VELOCITY:
    #         self.velocity_x = (
    #             self.velocity_x/abs(self.velocity_x)
    #             )*self.MAX_VELOCITY
    #     if abs(self.velocity_y) > self.MAX_VELOCITY:
    #         self.velocity_y = (
    #             self.velocity_y/abs(self.velocity_y)
    #             )*self.MAX_VELOCITY

    def score(self):
        return self.food_eaten
        # final_dist_target = np.hypot(
        #     self.center_x - self.target.center_x,
        #     self.center_y - self.target.center_y
        #     )
        # return self.init_dist_target/final_dist_target
        # return 1.0/(1 + np.hypot(
        #     food.center_x-self.center_x,
        #     food.center_y-self.center_y))


class Food(object):
    """ Represents a brick in my brick breaker game """
    def __init__(self, center_x, center_y, radius):
        """ Initializes a Brick object with the specified
            geometry and color """
        self.center_x = center_x
        self.center_y = center_y
        self.radius = radius
        # self.color = color
        self.eaten = False


class PyGameKeyboardController(object):
    def __init__(self, model):
        self.model = model

    def handle_event(self, event):
        """ Look for left and right keypresses to
            modify the x position of the paddle """
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
            time.sleep(.01)
        

    # nn = NN()
    # z1 = np.array([-1, 1])
    # print nn.process(z1)
