"""
The main code for the back propagation assignment. See README.md for details.
"""
import math
import copy
from typing import List

import numpy as np
from scipy.special import expit

class SimpleNetwork:
    """A simple feedforward network where all units have sigmoid activation.
    """

    @classmethod
    def random(cls, *layer_units: int):
        """Creates a feedforward neural network with the given number of units
        for each layer.

        :param layer_units: Number of units for each layer
        :return: the neural network
        """

        def uniform(n_in, n_out):
            epsilon = math.sqrt(6) / math.sqrt(n_in + n_out)
            return np.random.uniform(-epsilon, +epsilon, size=(n_in, n_out))

        pairs = zip(layer_units, layer_units[1:])
        return cls(*[uniform(i, o) for i, o in pairs])

    def __init__(self, *layer_weights: np.ndarray):
        """Creates a neural network from a list of weight matrices.
        The weights correspond to transformations from one layer to the next, so
        the number of layers is equal to one more than the number of weight
        matrices.

        :param layer_weights: A list of weight matrices
        """
        self.layer_weights = layer_weights

    def predict(self, input_matrix: np.ndarray) -> np.ndarray:
        """Performs forward propagation over the neural network starting with
        the given input matrix.

        Each unit's output should be calculated by taking a weighted sum of its
        inputs (using the appropriate weight matrix) and passing the result of
        that sum through a logistic sigmoid activation function.

        :param input_matrix: The matrix of inputs to the network, where each
        row in the matrix represents an instance for which the neural network
        should make a prediction
        :return: A matrix of predictions, where each row is the predicted
        outputs - each in the range (0, 1) - for the corresponding row in the
        input matrix.
        """
        # current activation. Start with the input matrix.
        a_i = copy.deepcopy(input_matrix)
        # list to store the activations. Store the input matrix as first element.
        a_l = [copy.deepcopy(input_matrix)]
        # list to store zs
        z_l = []
        # Apply each set of weights to the corresponding layer using the dot
            # product and apply the sigmoid transformation. Store each activation
            # and each z for later use.
            # source: https://github.com/mnielsen/neural-networks-and-deep-
                # learning/blob/master/src/network.py
        for weights in self.layer_weights:
            z_i = a_i.dot(weights)
            a_i = expit(z_i)
            z_l.append(z_i)
            a_l.append(a_i)

        return [a_i, a_l, z_l]

    def predict_zero_one(self, input_matrix: np.ndarray) -> np.ndarray:
        """Performs forward propagation over the neural network starting with
        the given input matrix, and converts the outputs to binary (0 or 1).

        Outputs will be converted to 0 if they are less than 0.5, and converted
        to 1 otherwise.

        :param input_matrix: The matrix of inputs to the network, where each
        row in the matrix represents an instance for which the neural network
        should make a prediction
        :return: A matrix of predictions, where each row is the predicted
        outputs - each either 0 or 1 - for the corresponding row in the input
        matrix.
        """
        # run the predict method to get initial predictions (belonging to a
            # logistic distribution)
        preds = self.predict(input_matrix)
        # return the version of the predictions that have been rounded to 0, 1
        return np.where(preds >= 0.5, 1, 0)

    def gradients(self,
                  input_matrix: np.ndarray,
                  output_matrix: np.ndarray) -> List[np.ndarray]:
        """Performs back-propagation to calculate the gradients for each of
        the weight matrices.

        This method first performs a pass of forward propagation through the
        network, then applies the following procedure to calculate the
        gradients. In the following description, × is matrix multiplication,
        ⊙ is element-wise product, and ⊤ is matrix transpose.

        First, calculate the error, error_L, between last layer's activations,
        h_L, and the output matrix, y:

        error_L = h_L - y

        Then, for each layer l in the network, starting with the layer before
        the output layer and working back to the first layer (the input matrix),
        calculate the gradient for the corresponding weight matrix as follows.
        First, calculate g_l as the element-wise product of the error for the
        next layer, error_{l+1}, and the sigmoid gradient of the next layer's
        weighted sum (before the activation function), a_{l+1}.

        g_l = (error_{l+1} ⊙ sigmoid'(a_{l+1}))⊤

        Then calculate the gradient matrix for layer l as the matrix
        multiplication of g_l and the layer's activations, h_l, divided by the
        number of input examples, N:

        grad_l = (g_l × h_l)⊤ / N

        Finally, calculate the error that should be backpropagated from layer l
        as the matrix multiplication of the weight matrix for layer l and g_l:

        error_l = (weights_l × g_l)⊤

        Once this procedure is complete for all layers, the grad_l matrices
        are the gradients that should be returned.

        :param input_matrix: The matrix of inputs to the network, where each
        row in the matrix represents an instance for which the neural network
        should make a prediction
        :param output_matrix: A matrix of expected outputs, where each row is
        the expected outputs - each either 0 or 1 - for the corresponding row in
        the input matrix.
        :return: two matrices of gradients, one for the input-to-hidden weights
        and one for the hidden-to-output weights
        """
        # do forward propogation and get predictions
        feed_forward = self.predict(input_matrix)
        # extract activations and pre-activation node values
        preds = feed_forward[0]
        a_l = feed_forward[1]
        z_l = feed_forward[2]
        # The error. Starting value is for last layer: predicted - observed
        error = preds - output_matrix
        # gradients for output
        gradients = []
        # define the sigmoid gradient
            # source: https://stackoverflow.com/a/27115201/9812619
        def sig_prime(array):
            return expit(array) * (1 - expit(array))
        # calculate gradient
            # z is calculated in the predict method, and is the pre-activation
            # weighted sums. self.z_l is reversed to facilitate more intuitive
            # iteration in the loop.
        for i in range(len(z_l)):
            # calculate g_l:
            g_l = (error * sig_prime(z_l[-i])).T
            # calculate grad_l over n input examples
            grad_l = g_l.dot(a_l[-i - 1]).T / input_matrix.shape[0]
            # store gradient matrix
            gradients.append(grad_l)
            # calculate error to backpropogate
            w_i = self.layer_weights[-i - 1]
            error = w_i.dot(g_l).T

        return gradients

    def train(self,
              input_matrix: np.ndarray,
              output_matrix: np.ndarray,
              iterations: int = 10,
              learning_rate: float = 0.1) -> None:
        """Trains the neural network on an input matrix and an expected output
        matrix.

        Training should repeatedly (`iterations` times) calculate the gradients,
        and update the model by subtracting the learning rate times the
        gradients from the model weight matrices.

        :param input_matrix: The matrix of inputs to the network, where each
        row in the matrix represents an instance for which the neural network
        should make a prediction
        :param output_matrix: A matrix of expected outputs, where each row is
        the expected outputs - each either 0 or 1 - for the corresponding row in
        the input matrix.
        :param iterations: The number of gradient descent steps to take.
        :param learning_rate: The size of gradient descent steps to take, a
        number that the gradients should be multiplied by before updating the
        model weights.
        """
