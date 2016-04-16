class NN(object):
    """Represents the Neural Network of a blob"""


    def __init__(self, parents_NN=None):
        """ 
        This neural network takes in difference three things:
        deltaX and deltaY between food and target,
        deltaAngle between food angle and target location


        parents_NN should be passed in as a tuple of NN objects
        """

        self.inputLayerSize = 3
        self.outputLayerSize = 5
        self.hiddenLayerSize = 6

        if parents_NN is not None:
            self.W1, self.W2 = self.get_recombine(parents_NN)
        else:
            self.W1 = np.random.uniform(-1, 1, (self.inputLayerSize, self.hiddenLayerSize))
            self.W2 = np.random.uniform(-1, 1, (self.hiddenLayerSize, self.outputLayerSize))


    def process(self, z1):
        """
        propigates the signal through the neural network.
        Return: 
        the decision that the blob will make (move forward, rotate counterclockwise, rotate clockwise,
        the magnitude of angle change (angular acceleration),
        the magnitude of distance change (velocity)
        """
        # input and output to level 2 (nodes)
        z2 = z1.dot(self.W1)
        a2 = self.sigmoid(z2)
        # input and output to level 3 (results)
        z3 = a2.dot(self.W2)
        a3 = self.sigmoid(z3)

        return [np.argmax(a3[0:2]), a3[3], a3[4]]


    def sigmoid(self, z):
        """
        Apply sigmoid activation function
        Note: this can currently overflow close to zero
        """

        return 1/(1+np.exp(-z))



class Blob(object):
    """ Represents a ball in my natural evolution simulation """


    def __init__(self, target, nn=None):
        """
        Create a ball object with the specified geometry 
        """
        self.center_x = random.randint(0, SCREEN_SIZE[0])
        self.center_y = random.randint(0, SCREEN_SIZE[1])
        self.int_center = int(self.center_x), int(self.center_y)
        self.angle = random.uniform(0,np.pi)
        self.target = target

        # Neural Network stuff here:
        if nn is not None:
            self.nn = nn
        else:
            self.nn = NN()


    def process_neural_net(self):
        """
        create environment and process through neural net brain
        """
        deltaX = self.target.center_x - self.center_x
        deltaY = self.target.center_y - self.center_y
        env = np.array([
            deltaX,
            deltaY,
            self.angle - np.arctan(deltaX/(deltaY+.000001))
            ])
        return self.nn.process(env)


    def decision_tree(self, decision, dist_mag, angle_mag):
        """
        modifies the position or angle based on neural net decision

        """
        if decision == 0: #move forward
            self.center_x += (1 + dist_mag)**2 * np.cos(self.angle)
            self.center_y += (1 + dist_mag)**2 * np.sin(self.angle)
            self.out_of_bounds()

            self.int_center = int(self.center_x), int(self.center_y)

        if decision == 1: #turn counter clockwise
            self.angle -= angle_mag

        if decision == 2: #turn clockwise
            self.angle += angle_mag


    def update(self, model):
        """ 
        Update the all aspects of blob based on neural net decisions
        """

        [decision, dist_mag, angle_mag] = self.process_neural_net()

        self.decision_tree(decision, dist_mag, angle_mag)

        self.target = model.foods[0]