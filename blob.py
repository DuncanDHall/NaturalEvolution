from constants import *
from food import *
from abstract import ParentSprite
import random
from nn import NN
import numpy as np
import math


class Blob(ParentSprite):
    """ Represents a ball in my natural evolution simulation """


    def __init__(self, target, nn=None):
        """ Create a ball object with the specified geometry """
        self.center_x = random.randint(0, SCREEN_SIZE[0])
        self.center_y = random.randint(0, SCREEN_SIZE[1])
        super(Blob, self).__init__(0, 0) #values are not needed
        self.int_center = int(self.center_x), int(self.center_y)
        self.radius = random.randint(5, 10)
        self.angle = random.uniform(0,np.pi)
        self.MAX_VELOCITY = 5
        self.energy = 1000
        self.MAX_ENERGY = 1000
        self.alive = True
        self.food_eaten = 0
        self.score_int = 0
        self.target = target
        #TODO: make these two part of the genes
        self.sight_radius = 150
        self.sight_angle = math.pi / 3

        #scoring related
        self.dist_moved = 0
        self.color = int(self.energy / 4 + 5)

        # Neural Network stuff here:
        if nn is not None:
            self.nn = nn
        else:
            self.nn = NN()

    def get_center_x(self):
        """Gets the x coordinate of the center"""
        #return a random value within constants screen size
        return self.center_x

    def get_center_y(self):
        """Gets the y coordinate of the center"""
        #return a random value within constants screen size
        return self.center_y

    def out_of_bounds(self):
        """ moves the blob to the other side of the screen if it moves out of 
            bounds.  It will also make sure angle is between 0 and 2pi 
        """
        # if self.center_x<0:
        #     self.center_x=int(SCREEN_SIZE[0])+self.center_x
        # if self.center_x>SCREEN_SIZE[0]:
        #     self.center_x=0+(self.center_x-int(SCREEN_SIZE[0]))

        # if self.center_y <0:
        #     self.center_y=int(SCREEN_SIZE[1])+self.center_y
        # if self.center_y>SCREEN_SIZE[1]:
        #     self.center_y=0+(self.center_y-int(SCREEN_SIZE[1]))

        if self.angle > 2*np.pi:
            self.angle = self.angle % np.pi
        if self.angle < -2*np.pi:
            self.angle = -self.angle % np.pi 


    def update_position(self, deltaDist):
        """ update_position based on an output from the neural net.  In
            addition, update attribute self.dist_moved for scoring related
            purposes
        """
        self.center_x += deltaDist * np.cos(self.angle)
        self.center_y += deltaDist * np.sin(self.angle)
        self.out_of_bounds()
        self.int_center = int(self.center_x), int(self.center_y)

    def update_angle(self, delta_angle):
        """ update_angle changes the angle based on an output from the neural
            net.
        """
        self.angle += delta_angle


    def process_neural_net(self):
        """ create environment and process through neural net brain
        """
        
        totalDistance = self.get_dist(self.target)
        energy_input = self.energy / 4. #scale engery to similar size.  Max input = 250
        change_angle = (SCREEN_SIZE[0]/2) * (self.angle - self.angle_between(self.target))

        env = np.array([
            totalDistance,
            change_angle,
            energy_input
            ])
        return self.nn.process(env)


    def update_energy(self, model, deltaDist, delta_angle):
        """ is_alive updates the energy of the blob based on a constant value.
            If the energy drops below zero, then remove the blob and add it
            score the model.vip_genes list.

            TODO: make blob lose energy based on distance moved
        """
        #subtract evergy based on distance moved
        self.energy -= np.abs(deltaDist) + 1
        if self.energy < 0:
            self.alive = False
            self.score_int = self.score()
            model.vip_genes.append((self.score_int, self.nn))

            model.blobs.remove(self)


    def get_things_within_sight(self, list_of_things):
        in_sight = []
        # closest_x = -1
        # closest_y = -1
        # closest_distance = 10000
        #iterate through all food
        for thing in list_of_things:
            x = thing.get_center_x()
            y = thing.get_center_y()
            distance = self.get_dist(thing)
            #checks if thing is within blob's radius of sight, and not right on top of it
            #right on top of itself is important when checking if other blobs are within sight
            if (distance < self.sight_radius and distance > 0):

                theta = self.angle_between(thing)
                #checks if food is within the blob's angle of sight
                if math.fabs(theta - self.angle) < self.sight_angle:
                    #within sight
                    in_sight.append(thing)
                    #if this is the closest food
                    # if distance < closest_distance:
                    #     closest_x = x
                    #     closest_y = y

        # if closest_x >= 0:
        #     return (closest_x, closest_y)
        # #if no food is found, return a random value to move towards
        # return (random.randint(0, 500), random.randint(0, 500))

        #return all the things in the list that are within sight
        return in_sight

    def eat_food(self, model):
        """ eat_food tests whether or not a blob eats food on a given frame.
            If a blobl eats food, add to the blobs energy and remove the food.
            In addition, asexually reproduce based on its neural net dna, and
            do some population control.

        """
        for i in range(len(model.foods)-1, -1, -1):
            f = model.foods[i]
            if self.intersect(f):
                self.food_eaten += 1
                self.energy += 500

                if self.energy > self.MAX_ENERGY:
                    self.energy = self.MAX_ENERGY

                del model.foods[i]
                
                model.foods.append(Food())

                model.blobs.append(Blob(model.foods[0], self.nn))

                if len(model.blobs) > BLOB_NUM:
                    energy_list = []
                    for blob in model.blobs:
                        energy_list.append(blob.energy)
                    del model.blobs[np.argmin(energy_list)]


    def score(self):
        """ score is the scoring / fitness function.  Try to make as simple as
            possible while still getting interesting behavior
        """
        return self.food_eaten

    def update_color(self):
        self.color = int(self.energy / 4 + 5)


    def update(self, model):
        """ Update the all aspects of blob based on neural net decisions. Also
            assign next food target.  

            TODO: make the food targeting a function.
        """

        [dist_mag, angle_mag] = self.process_neural_net()

        self.update_angle(angle_mag)

        self.update_position(dist_mag)

        self.update_energy(model, dist_mag, angle_mag)

        self.update_color()

        self.eat_food(model)

        #define target
        foods_in_sight = self.get_things_within_sight(model.foods)
        if len(foods_in_sight) > 0:
            self.target = foods_in_sight[0]
        else:
            #random point on screen
            #not sure if x corresponds to screen_size[0] and y to screen_size[1]
            randx = (self.center_x+random.randint(0, SCREEN_SIZE[0]))%SCREEN_SIZE[0]
            randy = (self.center_y+random.randint(0, SCREEN_SIZE[1]))%SCREEN_SIZE[1]
            self.target = ParentSprite(randx, randy)