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
        self.angle = np.pi / 2. #random.uniform(0,np.pi)
        self.energy = 1000
        self.MAX_ENERGY = 1000
        self.alive = True
        self.food_eaten = 0
        self.score_int = 0

        self.target_blob = self
        self.target_food = target

        #testing
        self.color = int(self.energy / 4 + 5)

        # Neural Network stuff here:
        if nn is not None:
            self.nn = NN(((1, nn),))
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
        # if self.center_x<0:
        #     self.center_x=int(SCREEN_SIZE[0])+self.center_x
        # if self.center_x>SCREEN_SIZE[0]:
        #     self.center_x=0+(self.center_x-int(SCREEN_SIZE[0]))

        # if self.center_y <0:
        #     self.center_y=int(SCREEN_SIZE[1])+self.center_y
        # if self.center_y>SCREEN_SIZE[1]:
        #     self.center_y=0+(self.center_y-int(SCREEN_SIZE[1]))

        if self.angle > 2*np.pi:
            self.angle = self.angle % (2 * np.pi)
        if self.angle < -2*np.pi:
            self.angle = -self.angle % (2 * np.pi)

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
        deltaXfood = self.target_food.center_x - self.center_x
        deltaYfood = self.target_food.center_y - self.center_y

        total_distance_food = np.hypot(deltaXfood, deltaYfood)

        change_angle_food = [(math.atan2(deltaYfood, deltaXfood) - self.angle),
                            (math.atan2(-1 * deltaYfood, deltaXfood) - self.angle)]
        #print change_angle_food
        change_angle_food_min_index = np.argmin(np.abs(change_angle_food))
        change_angle_food = change_angle_food[change_angle_food_min_index]
                            #* (SCREEN_SIZE[0]/2)
        energy_input = self.energy / 4. #scale engery to similar size.  Max input = 250

        #print deltaXfood, deltaYfood
        #print math.atan2(deltaYfood, deltaXfood)
        #print "nn input = " + str(change_angle_food * (180 / np.pi))
        #print self.angle * (180 / np.pi)

        deltaXblob = self.target_food.center_x - self.center_x
        deltaYblob = self.target_food.center_y - self.center_y
        total_distance_blob = np.hypot(deltaXblob, deltaYblob)
        change_angle_blob = (SCREEN_SIZE[0]/2) * (self.angle - math.atan2(deltaYblob, deltaXblob))

        env = np.array([
            total_distance_food,
            change_angle_food,
            energy_input,
            total_distance_blob,
            change_angle_blob
            ])
        return self.nn.process(env)


    def update_energy(self, model, deltaDist, deltaAngle):
        """ is_alive updates the energy of the blob based on a constant value.
            If the energy drops below zero, then remove the blob and add it
            score the model.vip_genes list.

            TODO: make blob lose energy based on distance moved
        """
        #subtract evergy based on distance moved
        self.energy -= np.abs(deltaDist) + .1
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
                self.energy += 500
                if self.energy > self.MAX_ENERGY:
                    self.energy = self.MAX_ENERGY

                del model.foods[i]

                # global SCREEN_SIZE
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


    def target_closest_blob(self, model):

        rel_blobs = [[np.hypot(self.center_x - blob.center_x, self.center_y - blob.center_y), blob] for blob in model.blobs]
        rel_blobs.sort()

        if len(rel_blobs) > 1:
            self.target_blob = rel_blobs[1][1] #get second smallest distance
        else:
            self.target_blob = self


    def target_closest_food(self, model):

        rel_food = [[np.hypot(self.center_x - food.center_x, self.center_y - food.center_y), food] for food in model.foods]
        rel_food.sort()
        self.target_food = rel_food[0][1]


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

        self.target_closest_blob(model)

        self.target_closest_food(model)
