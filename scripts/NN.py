import numpy as np
from scripts import functions as f

class DenseLayer:
    """
    This is a class that represents a fully connected layer in a neural network. A list of these layers defines an architecture for a "NeuralNetwork" object 
    defined later in this script.
    """
    def __init__(self, input_size, layer_size, activation_function="sigmoid", regularization="None", lamba=.2):
        """
        Initializes a DenseLayer object.

        Arguments:
            input_size::int
                Defines the number of incoming weights for each node in this layer. This will also be equal to the number of nodes in the previous layer,
                or the size of the input vector if this is the input layer. 
            layer_size::int
                Number of nodes in this layer. 
            activation_function::str (optional)
                String defining the non-linear activation function to be used in the nodes of this layer. The options in this implemenation are "sigmoid" and "relu".
                If not supplied, defaults to "sigmoid"
            regularization::str (optional)
                String defining the regularization function to be used on the outgoing weights of this layer. The options in this implementation are "None", "L2", and "Dropout".
                If not supplied, defaults to "None"
            lambda::float (optional)
                Regularization parameter that weights the regularization if using L2, or defines the dropout rate if using dropout. If not supplied, defaults to .2

        Returns:
            None     
        """
        self.input_size = input_size
        self.output_size = layer_size
        
        if activation_function == "sigmoid":
            self.activation = f.sigmoid
            self.activation_prime = f.sigmoid_prime
        elif activation_function == "relu":
            self.activation = f.relu
            self.activation_prime = f.relu_prime
        
        if regularization == "None":
            self.regularization = "None"
        elif regularization == "L2":
            self.regularization = "L2"
        elif regularization == "Dropout":
            self.regularization = "Dropout"

        self.weights = None
        self.biases = None
        self.lamba = lamba

    
    def feedforward(self, input, predict=False):
        """
        Performs the feedforward steps for this specific layer of the neural network. In other words, takes an input and re-weights it by the learned weights for this layer, and applies 
        the learned bias, followed by activation. Then passes this re-weighted vector as an output.

        Arguments:
            input::np.array(float)
                Array of input values to this layer. 
            predict::bool
                Flag for whether this function is being used in training or prediction. If prediction, we don't want to apply regularization (if regularization is used)
        
        Returns:
            output::np.array(float)
                Array of output values for this layer after re-weighting, bias adjustment, and activation.
        """
        self.input = input
        self.nonactivated = np.dot(input, self.weights) + self.biases
        output = self.activation(self.nonactivated)

        if self.regularization == "Dropout" and not predict:
            self.dropout_mask = np.random.binomial(1,self.lamba,size=output.shape) / self.lamba
            output = output * self.dropout_mask

        return output

    
    def backprop(self, output_error, learning_rate):
        """
        Performs the backpropogation steps for this specific layer of the neural network. In other words, takes a loss from the previous layer (or the derivative of the loss
        of a training step), and applies the chain rule to the parameters with respect to the loss function.
        Arguments:
            output_error::np.array(float)
                Array of losses of the previous layer 
            learning_rate::float
                Hyperparameter for how much the weights are adjusted in each backpropogation. 
        Returns:
            backpropogate_error::np.array(float)
                Array of loss to be backpropogated to the next layer in the network. 
        """

        # get the backpropogation from the activation function. This is also how much the biases need to be changed
        output_error = output_error * self.activation_prime(self.nonactivated)
        weights_delta = np.dot(self.input.T, output_error)
        
        # if we are using a regularization type, do it here
        if self.regularization == "Dropout":
            weights_delta = weights_delta * self.dropout_mask
        elif self.regularization == "L2":
            weights_delta = weights_delta + (self.lamba * self.weights)
        
        # adjust the weights and biases as needed
        self.weights -= learning_rate * weights_delta
        self.biases -= learning_rate * output_error

        # compute the vector that will be backproped to the next(previous?) layer
        backpropogate_error = np.dot(output_error, self.weights.T)

        return backpropogate_error


class NeuralNetwork:
    """
    This is a class that performs all of the functions of a basic neural network. A neural network is simply a series of layers that transform a numeric input into 
    an output of certain shape. These transformations can be learned to optimize a loss function, and can therefore be used to learn complex, non-linear functions. 
    A lot of the methods in this particular class are simply wrappers of methods of individual layers, from the DenseLayer class. I designed it this way intentionally to break 
    the neural network and its methods into modular components that can be easily understood on their own. 
    """
    def __init__(self, architecture=[DenseLayer(68,25,f.sigmoid), DenseLayer(25,1,f.sigmoid)],lr=.05,seed=1,loss_function = "mse"):
        """
        Initializes a NeuralNetwork object.

        Arguments:
            archtecture::[DenseLayer()]
                List of DenseLayer objects that together make up the design of the network. Each layer has an input shape, layer size, and activation function assigned to it.
                Regularization can be optionally added on a layer by layer basis
            lr::float
                Learning rate, which determines how much weights are adjusted during backpropogation. This param should be tuned
            seed::int
                Seed for random function seeding
            loss_function::str
                Loss function that the network optimizes with respect to. The available options in my implementation are mean-squared error, passed as "mse", or 
                binary cross-entrpy, passed as "bce". 
        
        Returns:
            None
        """
        self.architecture = architecture
        self.lr = lr
        np.random.seed(seed)
        self._make_weights()
        if loss_function == "mse":
            self.loss = f.mse
            self.loss_prime = f.mse_prime
        elif loss_function == "bce":
            self.loss = f.binary_cross_entropy
            self.loss_prime = f.binary_cross_entropy_prime


    def _make_weights(self):
        """
        Initializes all of the weights and biases in the neural network to random values between -1 and 1
        
        Arguments:
            None
        
        Returns:
            None
        """

        # for each layer, generate a set of weights and biases (initialized between -1:1) connecting each node in one layer with every node in the other layers
        for layer in self.architecture:
            layer.weights = (np.random.uniform(-1, 1, size = (layer.input_size, layer.output_size)))
            layer.biases = (np.random.uniform(-1, 1, size = (1, layer.output_size)))


    def feedforward(self, input, predict=False):
        """
        Takes an input vector and feeds it through the neural net to get it's corresponding output. If the neural net is thought of as a complex function,
        this is the equivalent of just calling the function on an input.

        Arguments:
            input::np.asarray(float)
                Input vector that is being fed through the network. Must be flat and of the same size as the input layer of the network
            predict::bool
                Flag to mark whether this method is being called during training or prediction. This affects whether weights are regularized. 
        
        Returns:
            output::np.asarray(float)
                Output of the neural net for the given input. Will be the size determined in the output layer of the network. 
        """
        # first check that the input shape matches the first layer of the network 
        if input.shape[1] != self.architecture[0].input_size:
            raise Exception("Input to feedforward step does not match the input layer")
        
        output = input
        for layer in self.architecture:
            output = layer.feedforward(output, predict)

        return output


    def backprop(self, error_prime):
        """
        Coordinates backpropogation between all of the layers in the neural network. Basically a wrapper for the individual DenseLayer classes held in 
        self.architecture.

        Arguments:
            error_prime::float
                Result of the derivative of the loss function applied to the output of the network compared to the true labels. Should be precalculated in
                self.fit()
        
        Returns:
            None
        """
        for layer in reversed(self.architecture):
            error_prime = layer.backprop(error_prime, self.lr)


    def fit(self, x, y, max_epochs=2000, mini_batch_negatives=False, print_losses=True):
        """
        Fits a neural network on the training examples provided in x, with their labels in y. 

        Arguments:
            x::np.asarray([int])
                List of observations that the network is trained on. Each observation should be a list holding a one-hot encoding of a DNA sequence
            y::np.asarray(int)
                Array holding the true labels for the observations in x
            max_epochs::int (optional)
                Number of epochs that the network will be trained.
            mini_batch_negatives::bool (optional)
                Flag telling whether or not the negative examples should be separately mini-batched. This feature is necessary because in the test-case 
                provided, the negative examples outweigh the positives by a factor of ~30,000.
            print_losses::bool (optional)
                Flag telling whether or not you want the training loss to be printed every 5 epochs.
        
        Returns:
            losses::[float]
                The average loss of the network for each training epoch
        """

        epoch_losses = []
        for i in range(max_epochs):
            losses = []
            if mini_batch_negatives:
                # reform the dataset so that it holds an equal number of positive and negative examples
                new_x = NeuralNetwork.mini_batch_negatives(x, y)
            else:
                new_x = x
            for sample, label in zip(new_x, y):

                sample = np.asarray([sample])
                # feed the sample through the neural net
                output = self.feedforward(sample)
                # compute the loss and print it
                loss = self.loss(label, output)
                losses.append(loss)

                # start backprop by calculating the derivative of loss function and calling backprop
                loss_prime = self.loss_prime(label, output)
                self.backprop(loss_prime)
            
            average_loss = sum(losses)/len(losses)
            epoch_losses.append(average_loss)
            # print the loss every 5 epoch (if specified)
            if (i+1) % 5 == 0 and print_losses:
                print("Epoch", i+1, "loss = ", str(average_loss))
            
        return epoch_losses
            

    def predict(self, x):
        """
        Generates a prediction from the network for a given input. This should only be called after fitting. 

        Arguments:
            x::np.asarray(float)
                Input vector that prediction is being generated for
        
        Returns:
            network_output::float
                Prediction for x
        """
        x = np.asarray([x])
        network_output = self.feedforward(x, predict=True)
        if network_output.shape == (1,1):
            network_output = network_output[0][0]
        else:
            network_output = network_output[0]
        return network_output
    
    @staticmethod
    def mini_batch_negatives(x, y):
        """
        Static function that performs mini-batching of the negative examples, for reasons described in the fit method documentation.

        Arguments:
            x::np.asarray([float])
                Full training set (includes both positive and negative examples)
            y::np.asarray(float)
                Labels for the training set stored in x
            
        Returns:
            batch::np.asarray([float])
                A batch that is twice the size of the number of positive examples in x (holds an equal number of positive and negative examples). 
                The negative examples in the batch are randomly sampled from the training set. 
        """

        # get pos and neg indices, and randomly sample from the negative inds
        pos_idx = (y==1).nonzero()[0]
        neg_idx = (y==0).nonzero()[0]
        random_neg_idx = np.random.choice(neg_idx.size, size=pos_idx.size, replace = False)

        # get the positive and negative examples to be used, and concatenate them into one array
        positive_examples = x[pos_idx]
        negative_examples = x[random_neg_idx]
        batch = np.concatenate([positive_examples, negative_examples])

        return batch