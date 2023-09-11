# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 10:29:54 2023

@author: Mayur
"""

import numpy as np

fname = 'data.csv'
data = np.genfromtxt(fname, dtype='float', delimiter=',', skip_header=1)
X, y = data[:, :-1], data[:, -1].astype(int)
X_train, y_train = X[:400], y[:400]
X_test, y_test = X[400:], y[400:]

Y = y.reshape(data.shape[0], 1)
Y.shape


# Dense Layers
class DenseLayer:
    def __init__(self, n_inputs, n_neurons):
        # Weights should be initialized with values drawn from a normal distribution scaled by 0.01.
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        # Biases are initialized to 0.0.
        # Biases are given to n_neurons which is the number of hidden layers. We do not provide for the input layers
        self.biases = np.zeros(n_neurons)

    def forward(self, inputs):
        self.inputs = inputs
        # The NP dot will create the dot prodicts with the input and weights which is the sum of products.
        # We then add biases to each of the new calculations
        self.z = np.dot(inputs, self.weights) + self.biases

    def backward(self, dz):
        # Gradients of weights
        self.dweights = np.dot(self.inputs.T, dz)
        # Gradients of biases
        self.dbiases = np.sum(dz, axis=0, keepdims=True)
        # Gradients of inputs
        self.dinputs = np.dot(dz, self.weights.T)

# Activations


class ReLu:
    """
    ReLu activation
    """

    def forward(self, z):
        """
        Forward pass
        """
        self.z = z
        # ReLu takes the max of 0 or z
        self.activity = np.maximum(0, z)

    def backward(self, dactivity):
        """
        Backward pass
        """
        self.dz = dactivity.copy()
        self.dz[self.z <= 0] = 0.0
        # DELETE LATER
        self.dinputs = self.dz


class Softmax:

    def forward(self, z):
        e_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        self.probs = e_z / e_z.sum(axis=1, keepdims=True)
        return self.probs

    def backward(self, dprobs):
        # Empty array
        self.dz = np.empty_like(dprobs)
        for i, (prob, dprob) in enumerate(zip(self.probs, dprobs)):
            # flatten to a column vector
            prob = prob.reshape(-1, 1)
            # Jacobian matrix
            jacobian = np.diagflat(prob) - np.dot(prob, prob.T)
            self.dz[i] = np.dot(jacobian, dprob)


# Loss Functions
class CrossEntropyLoss:
    def forward(self, probs, oh_y_true):
        """
        Use one-hot encoded y_true.
        """
        # clip to prevent division by 0
        # clip both sides to not bias up.
        probs_clipped = np.clip(probs, 1e-7, 1 - 1e-7)
        # negative log likelihoods
        loss = -np.sum(oh_y_true * np.log(probs_clipped), axis=1).mean(axis=0)
        return loss

    def backward(self, probs, oh_y_true):
        """
        Use one-hot encoded y_true.
        """
        # Number of examples in batch and number of classes
        batch_sz, n_class = probs.shape
        # get the gradient
        self.dprobs = -oh_y_true / probs
        # normalize the gradient
        self.dprobs = self.dprobs / batch_sz


# SGD Optimizer
class SGD:

    def __init__(self, learning_rate=0.5):
        # Initialize the optimizer with a learning rate
        self.learning_rate = learning_rate

    def update_params(self, layer):
        # Following slide 40 in lecture 2 of MMAI5500
        # weights = weights - learning rate* dweights
        # biases = biases - learning rate* dbiases
        # the weights and the biases come from doing backpropagation
        layer.weights = layer.weights - (self.learning_rate * layer.dweights)
        layer.biases = layer.biases - (self.learning_rate * layer.dbiases)


# Helper Functions
# Convert Probabilities to Predictions
def predictions(probs):
    """
    """
    y_preds = np.argmax(probs, axis=1)
    return y_preds


# Accuracy
def accuracy(y_preds, y_true):
    """
    """
    return np.mean(y_preds == y_true)


# One-hot encoding
oh_y_true = np.eye(3)[y_test]


# Training
# A single forward pass through the entire network
def forward_pass(X, y_true, oh_y_true):
    """  
    """
    dense1.forward(X)
    # This adds the activation function to the dotproduct of weights and inputs plus the bias
    activation1.forward(dense1.z)

    # This gives us y_hat which is the 4 hidden nodes in the second layer
    # we now need to pass it one more time to go to the output layer
    # We pass on the first activation function that comes from ReLu
    dense2.forward(activation1.activity)
    activation2.forward(dense2.z)

    # We now do it again for the second hidden layer with 8 nodes
    # We pass on the first activation function that comes from ReLu
    dense3.forward(activation2.activity)
    # We did two ReLu activations, now we must do a softmax activation for the output layer
    # We use the softmax class defined above which will return the probability
    probs = output_activation.forward(dense3.z)
    # Now we need the loss
    # For this we use the crossentropyloss class defined above
    # We compare with the probability output from the softmax function and the one hot encoded ground truth
    loss = crossentropy.forward(probs, oh_y_true)

    return probs, loss


# A single backward pass through the entire network

def backward_pass(probs, y_true, oh_y_true):

    # Gets the gradient with respect to output layer activations
    crossentropy.backward(probs, oh_y_true)
    # Backward pass through the softmax layer
    output_activation.backward(crossentropy.dprobs)
    dense3.backward(output_activation.dz)
    activation2.backward(dense3.dinputs)
    # Compute the gradient of the loiss with respect to pre-activaiton values of output layer
    dense2.backward(activation2.dz)
    # backpropagate through the second hidden layer
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dz)
    # backward pass through the first hidden layer


# Initialize network and set hyperparameters

# Initialize the network and set hyperparameters For example, number of
# epochs to train, batch size, number of neurons, etc.

# We set epochs to 100
epochs = 10
# Set batch size to 32
batch_sz = 10

n_batch = int(len(X_train)/batch_sz)
output_size = 3
n_class = 3
learning_rate = 0.5

dense1 = DenseLayer(3, 4)
activation1 = ReLu()
dense2 = DenseLayer(4, 8)
activation2 = ReLu()
dense3 = DenseLayer(8, 3)
output_activation = Softmax()
crossentropy = CrossEntropyLoss()
optimizer = SGD()

# Training Loop

for epoch in range(epochs):
    print('epoch:', epoch)
    for batch_i in range(n_batch):
        # Get a mini-batch of data from X_train and y_train. It should have batch_sz example"... YOUR CODE HERE ..."
        batch_X = X_train[batch_i * batch_sz: (batch_i + 1) * batch_sz]
        batch_y = y_train[batch_i * batch_sz: (batch_i + 1) * batch_sz]

        # One-hot encode y_true
        oh_y_true = np.eye(output_size)[batch_y]
        # Forward pass
        probs, loss = forward_pass(batch_X, batch_y, oh_y_true)
        # Print accuracy and loss
        y_preds = predictions(probs)
        acc = accuracy(y_preds, batch_y)
        print("Batch", batch_i+1, "Loss:", loss, "Accuracy:", acc)
        # Backward pass
        backward_pass(probs, batch_y, oh_y_true)
        # Update the weights
        optimizer.update_params(dense3)
        optimizer.update_params(dense2)
        optimizer.update_params(dense1)

# To test the code

probs, loss = forward_pass(X_test, y_test, np.eye(n_class)[y_test])
y_preds = predictions(probs)
acc = accuracy(y_preds, y_test)
print("Loss:", loss, "Accuracy:", acc)
