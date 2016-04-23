from constants import *
from food import *
import random
from nn import NN
import numpy as np
import math


class Blob(object):
    """ Represents a ball in my natural evolution simulation """


    def __init__(self, target, nn=None):
        """ Create a ball object with the specified geometry """
        self.center_x = random.randint(0, SCREEN_SIZE[0])
        self.center_y = random.randint(0, SCREEN_SIZE[1])
        self.int_center = int(self.center_x), int(self.center_y)
        self.radius = random.randint(5, 10)
        self.angle = random.uniform(0,np.pi)
        self.MAX_VELOCITY = 5
        self.energy = 100
        self.MAX_ENERGY = 200
        self.alive = True
        self.food_eaten = 0
        self.score_int = 0
        self.target = target
        #TODO: make these two part of the genes
        self.vision_mag = random.randint(100, 200)
        self.vision_angle = random.random()*math.pi/2.0

        #scoring related
        self.dist_moved = 0

        # Neural Network stuff here:
        if nn is not None:
            self.nn = nn
        else:
            self.nn = NN()


    def intersect(self, other):
        """ tells whether or not two objects are intersecting.  
            This will primarily be used to determine if a blob eats food
        """
        dist = abs(np.hypot(
            self.center_x-other.center_x, self.center_y-other.center_y))
        return dist < self.radius + other.radius


    def out_of_bounds(self):
        """ moves the blob to the other side of the screen if it moves out of 
            bounds.  It will also make sure angle is between 0 and 2pi 
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
        """ update_position based on an output from the neural net.  In
            addition, update attribute self.dist_moved for scoring related
            purposes
        """
        self.center_x += (1 + deltaDist)**2 * np.cos(self.angle)
        self.center_y += (1 + deltaDist)**2 * np.sin(self.angle)
        self.out_of_bounds()
        self.int_center = int(self.center_x), int(self.center_y)

        #update scoring
        self.dist_moved += deltaDist


    def update_angle(self, delta_angle):
        """ update_angle changes the angle based on an output from the neural
            net.
        """
        self.angle += delta_angle


    def process_neural_net(self, model):
        """ create environment and process through neural net brain
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


    def is_alive(self, model):
        """ is_alive updates the energy of the blob based on a constant value.
            If the energy drops below zero, then remove the blob and add it
            score the model.vip_genes list.

            TODO: make blob lose energy based on distance moved
        """
        self.energy -= .1
        if self.energy < 0:
            self.alive = False
            self.score_int = self.score()
            model.vip_genes.append((self.score_int, self.nn))

            model.blobs.remove(self)


    def eat_food(self, model):
        """ eat_food tests whether or not a blob eats food on a given frame.
            If a blobl eats food, add to the blobs energy and remove the food.
            In addition, asexually reproduce based on its neural net dna, and
            do some population control.

        """
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
        """ score is the scoring / fitness function.  Try to make as simple as
            possible while still getting interesting behavior
        """
        return self.dist_moved * (1 + self.food_eaten)


    def update(self, model):
        """ Update the all aspects of blob based on neural net decisions. Also
            assign next food target.  

            TODO: make the food targeting a function.
        """

        [dist_mag, angle_mag] = self.process_neural_net()

        self.update_angle(angle_mag)

        self.update_position(dist_mag)

        self.is_alive(model)

        self.eat_food(model)

        self.target = model.foods[0]