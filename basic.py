import pygame, math
import random
from pygame.locals import QUIT, KEYDOWN
import time
from random import choice
import numpy as np

screen_size = (500, 500)
sim_num = 0
blob_num = 10
sim_skip_num = 100 #the number of simulations you want to skip

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

        screen_size = self.screen.get_size()

        basicfont = pygame.font.SysFont(None, 48)
        sim_num_string = basicfont.render(str(sim_num), True, (255, 255, 255))
        self.screen.blit(sim_num_string, (1,1))

        for blob in self.model.blobs:
            
            if blob.alive:
                #print blob.int_center
                pygame.draw.circle(
                    self.screen, 
                    pygame.Color('white'),
                    blob.int_center,
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
        self.DNAresults = []
        #create foods
        for i in range(0, 1):
            x = screen_size[0]/2 #random.randint(0, screen_size[0])
            y = screen_size[1]/2 #random.randint(0, screen_size[1])
            radius = random.randint(5, 10)
            self.foods.append(Food(x, y, radius))
        #create blobs
        for i in range(0, blob_num):
            x = random.randint(0, screen_size[0])
            y = random.randint(0, screen_size[1])
            matrix = np.random.uniform(-.0001,.0001,(2,4))
            self.blobs.append(Blob(x, y, 10, matrix, self.foods[0]))

        # multiplies positions vector by DNA to produce velocity
        # changes self.vele[1])

        
    def update(self):
        """ Update the model state """
        for i in range(len(self.blobs)-1, -1, -1):
            self.blobs[i].update(self)

        #If all blobs are dead, start new cycle
        if len(self.blobs) <= 0:
            # num_winners = int( blob_num/5)

            # if num_winners%2 != 0:
            #     num_winners+=1
            # if num_winners<2:
            #     num_winners=2

            self.gene_recombination()
            self.DNAresults = []
            global sim_num
            sim_num+=1
            if sim_num % 10 == 0:
                print 'generation {} complete'.format(sim_num)
            
    def gene_recombination(self, num_winners=2):
        """Handles gene mutation and recombination"""
        top_scoring = sorted(self.DNAresults, key=lambda x:x[1])[-num_winners:]

        #average_dna = (1/2.) * np.add(top_two[0][0], top_two[1][0])
        average_dna = np.zeros(top_scoring[0][0].shape)

        #take a random gene from one of the parents
        for i in range(average_dna.shape[0]):
            for j in range( average_dna.shape[1]):
                average_dna[i][j]=random.choice([top_scoring[0][0][i][j], 
                    top_scoring[1][0][i][j]])

        #mutate one, make a new blob
        for i in range(0, blob_num):
            x = random.randint(0, screen_size[0])
            y = random.randint(0, screen_size[1])
            mutated_dna = np.copy(average_dna)
            if random.random()<.6: #mutation chance for altering a gene
                mutation = (random.random()-.5)*2*(10**-7)
                mutated_dna[random.randrange(average_dna.shape[0])][random.randrange(average_dna.shape[1])] += mutation
            elif random.random()<.05: #mutation chance for replacing a gene
                mutation = (random.random()-.5)*2*(10**-5)
                mutated_dna[random.randrange(average_dna.shape[0])][random.randrange(average_dna.shape[1])] = mutation
            self.blobs.append(Blob(x, y, 10, mutated_dna, self.foods[0]))
 
class Blob(object):
    """ Represents a ball in my brick breaker game """
    def __init__(self, center_x, center_y, radius, dna, target):
        """ Create a ball object with the specified geometry """
        self.center_x = center_x
        self.center_y = center_y
        self.int_center = int(self.center_x), int(self.center_y)
        self.radius = radius
        self.velocity_x = 0         # pixels / frame
        self.velocity_y = 0         # pixels / frame
        self.MAX_VELOCITY = 100
        self.energy = 100
        self.MAX_ENERGY = 100
        self.alive = True
        self.food_eaten = 0
        self.score_int = 0
        self.DNA = dna  # TODO don't hardcode this

        self.target = target
        self.init_dist_target = np.hypot(
            self.center_x - self.target.center_x,
            self.center_y - self.target.center_y
            )
        # self.init_dist_food = np.hypot()

    def intersect(self, other): 
        """
        Requires both objects to have center_x, center_y, and radius attributes
        """
        dist = abs(math.hypot(self.center_x-other.center_x, self.center_y-other.center_y))
        return dist < self.radius + other.radius

    def update(self, model, printvel = False):
        """ Update the position of the ball due to time passing """
        self.center_x += self.velocity_x
        self.center_y += self.velocity_y
        self.int_center = int(self.center_x), int(self.center_y)


        #print (self.center_x, self.center_y)

        # if self.center_x<0:
        #     self.center_x=0
        # if self.center_x>screen_size[0]:
        #     self.center_x=int(screen_size[0])

        # if self.center_y <0:
        #     self.center_y=0
        # if self.center_y>screen_size[1]:
        #     self.center_y=int(screen_size[1])

        self.energy -= .1
        if self.energy < 0:
            self.alive=False
            self.score_int = self.score()
            model.DNAresults.append((self.DNA, self.score_int))

            model.blobs.remove(self)

        # for i in range(len(model.foods)-1, -1, -1):
        #     food = model.foods[i]
        #     if self.intersect(food):
        #         self.food_eaten +=1

        #         model.foods.remove(food)

        #         x = random.randint(0, 500)
        #         y = random.randint(0, 500)
        #         radius = random.randint(5, 10)
        #         model.foods.append(Food(x, y, radius))

        self.change_vel()

    def change_vel(self, printvel = False): 
        positions = np.array([
            self.center_x, self.center_y,
            self.target.center_x, self.target.center_y])
        acceleration_x, acceleration_y = tuple(self.DNA.dot(positions))

        self.velocity_x += acceleration_x
        self.velocity_y += acceleration_y
        
        if abs(self.velocity_x)>self.MAX_VELOCITY:
            self.velocity_x = (self.velocity_x/abs(self.velocity_x))*self.MAX_VELOCITY
        if abs(self.velocity_y)>self.MAX_VELOCITY:
            self.velocity_y = (self.velocity_y/abs(self.velocity_y))*self.MAX_VELOCITY

        #print self.velocity_x, self.velocity_y
        # multiplies positions vector by DNA to produce velocity
        # changes self.vel
      
    def score(self):
        final_dist_target = np.hypot(
            self.center_x - self.target.center_x,
            self.center_y - self.target.center_y
            )
        return self.init_dist_target/final_dist_target
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
            for tup in sorted(model.DNAresults, key=lambda x:x[1])[-2:]:
                print tup
        if event.key == pygame.K_k:
            for blob in model.blobs:
                blob.energy = 0
        return True

if __name__ == '__main__':
    pygame.init()
    size = screen_size

    model = Model(size[0], size[1])
    view = PyGameBrickView(model, size)
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
        if sim_num % sim_skip_num == 0:
            view.draw()


