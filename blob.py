from constants import *
from food import *
from abstract import ParentSprite
from nn import NN
import numpy as np
import math


class Blob(ParentSprite):
    """ Represents a blob in the natural/articifial evolution simulation
    """


    def __init__(self, nn=None):
        """ 
        Initialize blob by inheriting ParentSprite and assigning attributes

        Args:
            nn (class): can pass in the neural net from another blob
        """
        super(Blob, self).__init__() #values are not needed
        self.int_center = int(self.center_x), int(self.center_y)
        self.radius = 10
        self.angle = random.uniform(0,2*np.pi)
        self.energy = MAX_ENERGY
        self.alive = True
        self.food_eaten = 0
        self.score_int = 0

        self.sight_angle = 10 * (np.pi / 180.)
        self.sight_radius = 1000

        self.target_blob = self
        self.target_food = self

        self.last_angle = .01

        #scoring related
        self.dist_moved = 0
        self.color = int(self.energy / 4 + 5)

        # Neural Network stuff here:
        if nn is not None:
            self.nn = NN(((1, nn),))
        else:
            self.nn = NN()

    def out_of_bounds(self):
        """ 
        updates self.angle so it is always between -2pi and +2pi
        """

        if self.angle > 2*np.pi:
            self.angle = self.angle % (2 * np.pi)
        if self.angle < -2*np.pi:
            self.angle = -self.angle % (2 * np.pi)


    def update_position(self, delta_dist):
        """ 
        updates self.center_x, self.center_y, and self.int_center

        Args:
            delta_dist (float): the distance output from the neural network
        """
        self.center_x += delta_dist * np.cos(self.angle)
        self.center_y += delta_dist * np.sin(self.angle)
        self.out_of_bounds()
        self.int_center = int(self.center_x), int(self.center_y)

    def update_angle(self, delta_angle):
        """ 
        updates self.angle based on the 
            net.
        """
        self.angle += delta_angle


    def process_neural_net(self):
        """ create environment and process through neural net brain
        """
        blob_target_input = 0 if self.target_blob == self else 1
        food_target_input = 0 if self.target_food == self else 1

        total_distance_food = self.get_dist(self.target_food)
        # change_angle_food = (SCREEN_SIZE[0]/2) * (self.angle - self.angle_between(self.target_food))
        energy_input = self.energy / 1000. #scale engery to similar size.  Max input = 250
        total_distance_blob = self.get_dist(self.target_blob)
        # change_angle_blob = (SCREEN_SIZE[0]/2) * (self.angle - self.angle_between(self.target_blob))

        env = np.array([
            blob_target_input,
            food_target_input,
            energy_input,
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


    def get_things_within_sight(self, list_of_things):
        in_sight = []
        # closest_x = -1
        # closest_y = -1
        # closest_distance = 10000
        #iterate through all food
        for thing in list_of_things:
            distance = self.get_dist(thing)
            #checks if thing is within blob's radius of sight, and not right on top of it
            #right on top of itself is important when checking if other blobs are within sight
            if distance > 0:
                theta = self.angle - self.angle_between(thing)
                theta = (theta + (2 * np.pi)) % (2 * np.pi)
                #checks if food is within the blob's angle of sight
                if np.fabs(theta) < self.sight_angle:
                    #within sight
                    in_sight.append(thing)

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

                if self.energy > MAX_ENERGY:
                    self.energy = MAX_ENERGY

                del model.foods[i]

                model.foods.append(Food())

                model.blobs.append(Blob(NN([(1, self.nn)])))

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


    def target_closest_blob(self, blob_list):

        rel_blobs = [[self.get_dist(blob), blob] for blob in blob_list]
        rel_blobs.sort()

        if len(rel_blobs) > 1:
            self.target_blob = rel_blobs[1][1] #get second smallest distance
        else:
            self.target_blob = self


    def target_closest_food(self, food_list):

        rel_food = [[self.get_dist(food), food] for food in food_list]
        rel_food.sort()

        if len(rel_food) > 0:
            self.target_food = rel_food[0][1]
        else:
            self.target_food = self

    def update(self, model):
        """ Update the all aspects of blob based on neural net decisions. Also
            assign next food target.

            TODO: make the food targeting a function.
        """

        [dist_mag, angle_mag] = self.process_neural_net()

        self.last_angle = angle_mag

        self.update_angle(angle_mag)

        self.update_position(dist_mag)

        self.update_energy(model, dist_mag, angle_mag)

        self.update_color()

        self.eat_food(model)

        self.target_closest_blob(self.get_things_within_sight(model.blobs))

        self.target_closest_food(self.get_things_within_sight(model.foods))
