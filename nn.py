import numpy as np
from constants import *
import random


class NN(object):
    """ Represents the Neural Network of a blob """
    def __init__(self, parents_NN=None):
        """ this neural network takes in difference in x and y position between
            the agent and a single food entity.
            parents_NN should be passed in as a tuple of NN objects
        """
        if parents_NN is not None:
            self.W1, self.W2 = self.get_recombine(parents_NN)

        else:
            self.W1 = np.random.uniform(-1, 1, (2, NUM_NODES))
            self.W2 = np.random.uniform(-1, 1, (NUM_NODES, 2))

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
        return a3

    def sigmoid(self, z):
        # Apply sigmoid activation function (arctan):
        sig = 10*(1/(1+np.exp(-z))-.5)
        return sig
