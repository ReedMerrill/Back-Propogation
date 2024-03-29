"""
The main code for the back propagation assignment. See README.md for details.
"""
import math
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

    def forward_prop(self, input_matrix: np.ndarray):
        """Do forward propogation, returning two versions of the activations
        for each layer: one that holds the node values before the sigmoid is
        applied (z_l), and one that stores all the actual activations (a_l).
        """
        # do forward propogation and get predictions
        # list to store zs
        z_l = []
        # list to store the activations
        a_l = [input_matrix]
        # source: https://github.com/mnielsen/neural-networks-and-deep-
            # learning/blob/master/src/network.py
        for i, weights in enumerate(self.layer_weights):
            # Apply each set of weights to the corresponding layer
            z_i = np.dot(a_l[i], weights)
            # apply the sigmoid activation function
            a_i = expit(z_i)
            # add pre-activation weighted sums and activations to their lists
            z_l.append(z_i)
            a_l.append(a_i)

        return [z_l, a_l]

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
        # call forward prop helper to get activations and final prediction
        activations = self.forward_prop(input_matrix)
        #extract activations
        a_l = activations[1]
        # return the last activation: the model predictions
        return a_l[-1]

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
        # call forward prop helper to get activations and final prediction
        activations = self.forward_prop(input_matrix)
        # extract node values before sigmoid is applied
        z_l = activations[0]
        # extract activations
        a_l = activations[1]

        # calculate the cost
        error = a_l[-1] - output_matrix

        # Do back-propogation to calculate gradients
        # initialize list of gradients
        gradients = []
        # reverse a_l and z_l to make indexing easier
        a_l_reversed = a_l[::-1]
        z_l_reversed = z_l[::-1]
        w_l_reversed = self.layer_weights[::-1]
        # calculate gradient
        for i in range(len(z_l)):
            # calculate g_l:
            g_l = (error * sig_prime(z_l_reversed[i])).T
            # dot product of g_l and current activation / n input examples
            grad_l = (np.dot(g_l, a_l_reversed[i + 1])).T / input_matrix.shape[0]
            # store gradient matrix
            gradients.append(grad_l)
            # calculate error to back-propogate
            error = (np.dot(w_l_reversed[i], g_l)).T

        # return gradients
        # reverse to change them to the order of the actual network
        return gradients[::-1]

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
        # perform "iterations" number of gradient steps
        # initialize counter variable
        i = 0
        while i != iterations:

            # calculate gradient matrices
            gradients = self.gradients(input_matrix, output_matrix)
            # list of updated weights
            updated_weights = []
            for g_i, w_i in zip(gradients, self.layer_weights):
                # apply gradient matrices to previous weights matrices
                new_weights = w_i - (learning_rate * g_i)
                # update weights
                updated_weights.append(new_weights)

            # increment counter
            i += 1

            # asign updated weights to self.layer weights to be used in next
                # gradient step
            self.layer_weights = updated_weights

# Helper Functions
def sig_prime(array):
    """Define sigmoid prime."""
    # source: https://stackoverflow.com/a/27115201/9812619
    return expit(array) * (1 - expit(array))
