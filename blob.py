from constants import *
from food import *
from abstract import ParentSprite
from nn import NN
import numpy as np
import math



class Blob(ParentSprite):
    """ 
    Represents a blob in the natural/articifial evolution simulation
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


    def update_position(self, velocity):
        """ 
        updates self.center_x, self.center_y, and self.int_center

        Args:
            velocity (float): the velocity output from the neural network
        """
        self.center_x += velocity * np.cos(self.angle)
        self.center_y += velocity * np.sin(self.angle)
        self.out_of_bounds()
        self.int_center = int(self.center_x), int(self.center_y)


    def update_angle(self, angular_velocity):
        """ 
        updates self.angle based on the angular velocity net.

        Args:
            angular_velocity (float): the angular velocity output from the
            neural network
        """
        self.angle += angular_velocity


    def process_neural_net(self):
        """
        use blob's neural network to determine velocity and angular velocity

        Returns:
            list containing distance and angle magnitudes
        """
        #assign binary inputs if the blob can see a food or blob object
        blob_target_input = 0 if self.target_blob == self else 1 
        food_target_input = 0 if self.target_food == self else 1

        #preprocess neural net inputs
        energy_input = self.energy / 1000. #scale engery between 1 through 0

        #create array containing neural net inputs
        env = np.array([
            blob_target_input,
            food_target_input,
            energy_input,
            ])
        return self.nn.process(env)


    def update_energy(self, model, velocity, angular_velocity):
        """ 
        updates self.energy of a blob based on the distance it moves and an 
        energy loss constant

        Args:
            model (object): contains attributes of the environment
            velocity (float): the distance the blob will move
            angular_velocity (float): the angle the blob will change
        """
        #subtract evergy based on distance moved
        self.energy -= np.abs(velocity) + ENERGY_LOSS_CONSTANT
        if self.energy < 0:
            self.alive = False
            self.score_int = self.score()
            model.vip_genes.append((self.score_int, self.nn))

            model.blobs.remove(self)


    def get_things_within_sight(self, list_of_things):
        """
        determines what objects are within a blob's field of vision

        Args:
            list_of_things (list): a list of objects created using ParentSprite class

        Returns:
            list containing objects in a blob's field of vision
        """
        in_sight = []

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
        """ 
        tests whether or not a blob eats food on a given frame. If a blob 
        eats food, remove the food, increase the blob's energy, asexually 
        reproduce based on its neural net dna, and do some population control.

        Args:
            model (object): contains attributes of the environment

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
        """
        gives a blob's score based on: self.food_eaten

        Returns:
            the score of the blob
        """
        return self.food_eaten


    def update_color(self):
        """
        update self.color on it's current energy
        """
        self.color = int(self.energy / 4 + 5)


    def target_closest_blob(self, blob_list):
        """
        update self.target_blob to the closest blob given a list of blobs. If 
        no blobs in sight, update self.target_blob to self

        Args:
            blob_list (list): list of blobs. Generally blobs it can see.
        """
        rel_blobs = [[self.get_dist(blob), blob] for blob in blob_list]
        rel_blobs.sort()

        if len(rel_blobs) > 1:
            self.target_blob = rel_blobs[1][1] #get second smallest distance
        else:
            self.target_blob = self


    def target_closest_food(self, food_list):
        """
        update self.target_food to the closest food given a list of foods. If 
        no food in sight, update self.target_food to self

        Args:
            food_list (list): list of foods. Generally foods it can see.
        """
        rel_food = [[self.get_dist(food), food] for food in food_list]
        rel_food.sort()

        if len(rel_food) > 0:
            self.target_food = rel_food[0][1]
        else:
            self.target_food = self


    def update(self, model):
        """ 
        Update the blob by calling helper functions.
        """
        # get current velocity and angular velocity based on neural network
        [velocity, angular_velocity] = self.process_neural_net()
        
        # use velocity and angular_velocity to update self.angle, 
        # self.center_x, self.center_y, and self.energy
        self.update_angle(angular_velocity)

        self.update_position(velocity)

        self.update_energy(model, velocity, angular_velocity)

        # update color of blob
        self.update_color()

        # interact with food objects
        self.eat_food(model)

        # re-assign targets
        self.target_closest_blob(self.get_things_within_sight(model.blobs))

        self.target_closest_food(self.get_things_within_sight(model.foods))


