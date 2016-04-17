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
SIM_SKIP_NUM = 10  # the number of simulations you want to skip
NUM_PARENTS = 2

MUTATION_RATE = 0.2
MUTATION_AMOUNT = 0.5

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

        self.show_gen = False

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

class NN(object):
    """ Represents the Neural Network of a blob """


    def __init__(self, parents_NN=None):
        """ this neural network takes in difference in x and y position between
            the agent and a single food entity.
            parents_NN should be passed in as a tuple of NN objects
        """

        self.inputLayerSize = 2
        self.outputLayerSize = 2
        self.hiddenLayerSize = 4

        if parents_NN is not None:
            self.W1, self.W2 = self.get_recombine(parents_NN)
        else:
            self.W1 = np.random.uniform(-1, 1, (self.inputLayerSize, self.hiddenLayerSize))
            self.W2 = np.random.uniform(-1, 1, (self.hiddenLayerSize, self.outputLayerSize))


    def get_recombine(self, parents_NN):
        new_W_list = []

        list_ws = [(n[1].W1, n[1].W2) for n in parents_NN]

        for W_parents in zip(*list_ws):
            dim = W_parents[0].shape

            for w_par in W_parents:
                if w_par.shape != dim:
                    raise ValueError
            new_W = np.zeros(dim)
            for r in range(dim[0]):
                for c in range(dim[1]):
                    new_W[r][c] = random.choice(
                        [n[r][c] for n in W_parents]) + \
                        self.get_mutation()
            new_W_list.append(new_W)
        return tuple(new_W_list)


    def get_mutation(self):
        if np.random.rand() < MUTATION_RATE:
            return np.random.uniform(-MUTATION_AMOUNT, MUTATION_AMOUNT)
        return 0


    def process(self, z1):
        """ propigates the signal through the neural network """
        # input and output to level 2 (nodes)
        z2 = z1.dot(self.W1)
        a2 = self.sigmoid(z2)
        # input and output to level 3 (results)
        z3 = a2.dot(self.W2)
        a3 = self.sigmoid(z3)

        return [a3[0], a3[1]]


    def sigmoid(self, z):
        # Apply sigmoid activation function
        return (1/(1+np.exp(-z))) - .5



class Blob(object):
    """ Represents a ball in my natural evolution simulation """
    def __init__(self, target, nn=None):
        """ Create a ball object with the specified geometry """
        self.center_x = random.randint(0, SCREEN_SIZE[0])
        self.center_y = random.randint(0, SCREEN_SIZE[1])
        self.int_center = int(self.center_x), int(self.center_y)
        self.radius = random.randint(10, 20)
        self.angle = random.uniform(0,np.pi)
        self.MAX_VELOCITY = 5
        self.energy = 100
        self.MAX_ENERGY = 200
        self.alive = True
        self.food_eaten = 0
        self.score_int = 0
        self.target = target

        #scoring related
        self.last_angle_sign = True
        self.num_opposite_spins = 0

        self.dist_moved = 0

        # Neural Network stuff here:
        if nn is not None:
            self.nn = nn
        else:
            self.nn = NN()


    def intersect(self, other):
        """ 
        tells whether or not two objects are intersecting.  This will
        primarily be used to determine if a blob eats food
        """
        dist = abs(math.hypot(
            self.center_x-other.center_x, self.center_y-other.center_y))
        return dist < self.radius + other.radius


    def out_of_bounds(self):
        """
        moves the blob to the other side of the screen 
        if it moves out of bounds.  It will also make sure angle is
        between 0 and 2pi 
        """
        if self.center_x<0:
            self.center_x=int(SCREEN_SIZE[0])+self.center_x
        if self.center_x>SCREEN_SIZE[0]:
            self.center_x=0+(self.center_x-int(SCREEN_SIZE[0]))

        if self.center_y <0:
            self.center_y=int(SCREEN_SIZE[1])+self.center_y
        if self.center_y>SCREEN_SIZE[1]:
            self.center_y=0+(self.center_y-int(SCREEN_SIZE[1]))

        if self.angle > 2*np.pi:
            self.angle = self.angle % np.pi       
        if self.angle < -2*np.pi:
            self.angle = -self.angle % np.pi 


    def update_position(self, deltaDist):
        self.center_x += (1 + deltaDist)**2 * np.cos(self.angle)
        self.center_y += (1 + deltaDist)**2 * np.sin(self.angle)
        self.out_of_bounds()
        self.int_center = int(self.center_x), int(self.center_y)

        #add dist_moved to score
        self.dist_moved += deltaDist


    def update_angle(self, delta_angle):
        new_angle = self.angle + delta_angle

        latest_sign = delta_angle > 0

        if latest_sign != self.last_angle_sign:
            self.num_opposite_spins += 1

        self.last_angle_sign = latest_sign
        self.angle = new_angle


    def process_neural_net(self):
        """
        create environment and process through neural net brain
        """
        deltaX = self.target.center_x - self.center_x
        deltaY = self.target.center_y - self.center_y
        totalDistance = np.hypot(deltaX, deltaY)
        changeAngle = self.angle -  np.arctan(deltaY/(deltaX+.000001))

        env = np.array([
            totalDistance,
            changeAngle
            ])
        return self.nn.process(env)


    def is_alive(self):
        self.energy -= .1
        if self.energy < 0:
            self.alive = False
            self.score_int = self.score()
            model.vip_genes.append((self.score_int, self.nn))

            model.blobs.remove(self)


    def eat_food(self, model):
        for i in range(len(model.foods)-1, -1, -1):
            f = model.foods[i]
            if self.intersect(f): #where is this global f defined
                self.food_eaten += 1
                self.energy += 50
                if self.energy > self.MAX_ENERGY:
                    self.energy = self.MAX_ENERGY

                del model.foods[i]

                # global SCREEN_SIZE
                model.foods.append(
                    Food(
                        random.randint(10, SCREEN_SIZE[0]-10),
                        random.randint(10, SCREEN_SIZE[1]-10),
                        random.randint(5, 10)))

                model.blobs.append(Blob(model.foods[0], self.nn))

                if len(model.blobs) > 10:
                    energy_list = []
                    for blob in model.blobs:
                        energy_list.append(blob.energy)
                    del model.blobs[np.argmin(energy_list)]



    def score(self):
        return self.dist_moved * self.num_opposite_spins * (.1 + self.food_eaten)


    def update(self, model):
        """ 
        Update the all aspects of blob based on neural net decisions
        """

        [dist_mag, angle_mag] = self.process_neural_net()

        self.update_angle(angle_mag)

        self.update_position(dist_mag)

        self.is_alive()

        self.eat_food(model)

        self.target = model.foods[0]


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
        if event.key == pygame.K_s:
            model.show_gen = not model.show_gen
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
        if model.show_gen:
            view.draw()
            time.sleep(.01)
        

    # nn = NN()
    # z1 = np.array([-1, 1])
    # print nn.process(z1)
