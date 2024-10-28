import numpy as np
import random

class Network(object):

    def __init__(self,sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y,1) for y in sizes[1:]] # dimension n x 1, excludes first (input) layer 
        self.weights = [np.random.randn(y,x) for x, y in zip(sizes[:-1], sizes[1:])] # matrix of connections, w_ij with j -> i

    def feedforward(self,a):
        """Return the output of the network given 'a' as an input"""
        for b,w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w,a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        """Trains the neutral network using minibatch stochastic gradient descent.
        Training data is a list of tuples (x,y) containing training inputs and 
        desired outputs"""
        if test_data: 
            n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0,n,mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch,eta)
            if test_data:
                print("Epoch {0}: {1} / {2}".format(j,self.evaluate(test_data),n_test))
            else:
                print("Epoch{0}complete".format(j))

    def update_mini_batch(self, mini_batch, eta):
        """
        Updates network weights and biases using SGD backpropagation to a single mini batch 
        'x' is training inputs
        'y' is desired outputs
        We do one training example (x,y) at a time 
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases] # initialise array
        nabla_w = [np.zeros(w.shape) for w in self.weights] # initialise array
        for x,y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x,y) 
            # update new cost gradient based on new training example
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            # now make a step to minimise cost
            self.weights = [w-(eta/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)]
            self.biases = [b-(eta/len(mini_batch))*nb for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """
        Returns tuple (nabla_b, nabla_w) which is the gradient for the cost function C_x
        'nabla_b' and 'nabla_w' are layer-by-layer lists of numpy arrays
        """
        # initialise arrays
        nabla_b = [np.zeros(b.shape) for b in self.biases] 
        nabla_w = [np.zeros(w.shape) for w in self.weights] 
        activation = x # training input
        activations = [x] 
        zs = []
        # feedforward #
        ###############
        for b,w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b # calculate all weighted inputs in the network, each row is a layer 
            zs.append(z)
            activation = sigmoid(z) # calculate all activations in the network
            activations.append(activation)
        # backward pass #
        #################
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1]) # the error of the output layer
        nabla_b[-1] = delta # third network equation
        nabla_w[-1] = np.dot(delta, activations[-2].transpose()) # fourth network equation 
        for l in range(2, self.num_layers): # start from l-1, move backwards through layers
            z = zs[-l] 
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp # second network equation
            # calculate cost gradient #
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose()) 
        return (nabla_b, nabla_w) # return cost gradient for all layers

    def evaluate(self,test_data):
        """
        Returns the number of test inputs for which the neural network is correct
        The neural network prediction is the output neuron with the highest activation
        """
        test_results = [(np.argmax(self.feedforward(x)), y) for (x,y) in test_data]
        return sum(int(x == y) for (x,y) in test_results)
    
    def cost_derivative(self, output_activations, y):
        """
        Return the vector of partial derivatives for the output activations
        The partial derivative of the quadratic cost function w.r.t. activation
        """
        return (output_activations - y)

# Misc functions #
##################
def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))