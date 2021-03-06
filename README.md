# Project 3 - Neural Networks
# Due 03/22/2021

![BuildStatus](https://github.com/rle1323/Project3/workflows/HW3/badge.svg?event=push)

## main
Runs all code in scripts/\_\_main\_\_.py, useful for part 2
```
python -m scripts
```

## testing
Testing is as simple as running
```
python -m pytest test/*
```
from the root directory of this project.

## API Documentation

The documentation for the API is broken into three parts: Documentation for the classes pertaining to the Neural Network implementation, documentation for utility functions contained in scripts/functions.py, and documentation for the data I/O functions made for this project. 

### Neural Network Documentation

A neural network can be initialized with a NeuralNetwork() class object, and it's architecture is defined as a list of DenseLayer objects. Further documentation for both of these classes and their methods are below. A NeuralNetwork object can be trained with the fit() method, and predictions can be made with the predict() method.

#### class NeuralNetwork:
This is a class that performs all of the functions of a basic neural network. A neural network is simply a series of layers that transform a numeric input into 
an output of certain shape. These transformations can be learned to optimize a loss function, and can therefore be used to learn complex, non-linear functions. 
A lot of the methods in this particular class are simply wrappers of methods of individual layers, from the DenseLayer class. I designed it this way intentionally to break the neural network and its methods into modular components that can be easily understood on their own. 

##### NeuralNetwork Class Methods

```
__init__(self, architecture=[DenseLayer(68,25,f.sigmoid), DenseLayer(25,1,f.sigmoid)],lr=.05,seed=1,loss_function = "mse"):
        """
        Initializes a NeuralNetwork object.

        Arguments:
            archtecture::[DenseLayer()]
                List of DenseLayer objects that together make up the design of the network. Each layer has an input shape, layer size, and activation function 
                assigned to it.
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


_make_weights(self):
        Initializes all of the weights and biases in the neural network to random values between -1 and 1
        
        Arguments:
            None
        
        Returns:
            None


feedforward(self, input, predict=False):
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


backprop(self, error_prime):
        Coordinates backpropogation between all of the layers in the neural network. Basically a wrapper for the individual DenseLayer classes held in 
        self.architecture.

        Arguments:
            error_prime::float
                Result of the derivative of the loss function applied to the output of the network compared to the true labels. Should be precalculated in
                self.fit()
        
        Returns:
            None


fit(self, x, y, max_epochs=2000, mini_batch_negatives=False, print_losses=True):
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


predict(self, x):
        Generates a prediction from the network for a given input. This should only be called after fitting. 

        Arguments:
            x::np.asarray(float)
                Input vector that prediction is being generated for
        
        Returns:
            network_output::float
                Prediction for x


mini_batch_negatives(x, y):
        Static method that performs mini-batching of the negative examples, for reasons described in the fit method documentation.

        Arguments:
            x::np.asarray([float])
                Full training set (includes both positive and negative examples)
            y::np.asarray(float)
                Labels for the training set stored in x
            
        Returns:
            batch::np.asarray([float])
                A batch that is twice the size of the number of positive examples in x (holds an equal number of positive and negative examples). 
                The negative examples in the batch are randomly sampled from the training set. 
```


#### class DenseLayer:
This is a class that represents a fully connected layer in a neural network. A list of these layers defines an architecture for a "NeuralNetwork" object.


##### DenseLayer Class Methods
```
__init__(self, input_size, layer_size, activation_function="sigmoid", regularization="None", lamba=.2):
        """
        Initializes a DenseLayer object.

        Arguments:
            input_size::int
                Defines the number of incoming weights for each node in this layer. This will also be equal to the number of nodes in the previous layer,
                or the size of the input vector if this is the input layer. 
            layer_size::int
                Number of nodes in this layer. 
            activation_function::str (optional)
                String defining the non-linear activation function to be used in the nodes of this layer. The options in this implemenation are "sigmoid" and 
                "relu".
                If not supplied, defaults to "sigmoid"
            regularization::str (optional)
                String defining the regularization function to be used on the outgoing weights of this layer. The options in this implementation are "None", "L2", 
                and "Dropout".
                If not supplied, defaults to "None"
            lambda::float (optional)
                Regularization parameter that weights the regularization if using L2, or defines the dropout rate if using dropout. If not supplied, defaults to .2

        Returns:
            None 
            
feedforward(self, input, predict=False):
        """
        Performs the feedforward steps for this specific layer of the neural network. In other words, takes an input and re-weights it by the learned weights for 
        this layer, and applies the learned bias, followed by activation. Then passes this re-weighted vector as an output.

        Arguments:
            input::np.array(float)
                Array of input values to this layer. 
            predict::bool
                Flag for whether this function is being used in training or prediction. If prediction, we don't want to apply regularization (if regularization is 
                used)
        
        Returns:
            output::np.array(float)
                Array of output values for this layer after re-weighting, bias adjustment, and activation.

backprop(self, output_error, learning_rate):
        Performs the backpropogation steps for this specific layer of the neural network. In other words, takes a loss from the previous layer (or the derivative 
        of the loss of a training step), and applies the chain rule to the parameters with respect to the loss function.
        Arguments:
            output_error::np.array(float)
                Array of losses of the previous layer 
            learning_rate::float
                Hyperparameter for how much the weights are adjusted in each backpropogation. 
        Returns:
            backpropogate_error::np.array(float)
                Array of loss to be backpropogated to the next layer in the network. 
```

### Utility Function Documentation

```
sigmoid(x):
    Expression that calculates the sigmoid of a number x
    This is used as a non-linear activation function.

    Arguments:
        x::float or np.array(float)
            Scalar or array of scalars that expression is performed on
    
    Returns::
        y::float or np.array(float)
            Result of sigmoid(x)


sigmoid_prime(x):
    Expression that calculates the derivative function of sigmoid of a number x
    The integral of this function is used as a non-linear activation function, and this derivative function is called in backpropogation
    
    Arguments:
        x::float or np.array(float)
            Scalar or array of scalars that expression is performed on
    
    Returns::
        y::float or np.array(float)
            Result of sigmoid(x)
            
            
relu(x):
    Expression that calculates the ReLU function of a scalar. ReLU(x) = x if x > 0, and 0 if x <= 0.
    This is used as a non-linear activation function.
    
    Arguments:
        x::float or np.array(float)
            Scalar or array of scalars that expression is performed on
    
    Returns::
        y::float or np.array(float)
            Result of relu(x)
            
            
relu_prime(x):
    Expression that calculates the derivative of the ReLU function of a scalar.
    The integral of this function is used as a non-linear activation function, and this derivative function is called in backpropogation
    
    Arguments:
        x::float or np.array(float)
            Scalar or array of scalars that expression is performed on
    
    Returns::
        y::float or np.array(float)
            Result of relu(x)
       
mse(y, y_hat):
    Calculates the mean-squared error between a prediction and true value

    Arguments:
        y::np.array(float)
            Ground truth labels
        y_hat::np.array(float) 
            Predicted labels
    
    Returns:
        mse::float
            Mean-squared error calculated
            
            
mse_prime(y, y_hat):
    Expression that calculates the derivative function of mse of a prediction, given the predicted and true values
    
    Arguments:
        y::np.array(float)
            Ground truth labels
        y_hat::np.array(float) 
            Predicted labels
    
    Returns:
        mse_prime::float
            Mean-squared prime calculated


binary_cross_entropy(y, y_hat):
    Calculates the mean-squared error between a prediction and true value. This function is used as a neural network loss function

    Arguments:
        y::np.array(float)
            Ground truth labels
        y_hat::np.array(float) 
            Predicted labels
    
    Returns:
        bce::float
            Binary cross-entropy calculated


binary_cross_entropy_prime(y, y_hat):
    Expression that calculates the derivative function of binary cross-entropy of a prediction, given the predicted and true values. 
    
    Arguments:
        y::np.array(float)
            Ground truth labels
        y_hat::np.array(float) 
            Predicted labels
    
    Returns:
        bce_prime::float
            Binary cross-entropy prime calculated


dna_onehot(dna):
    One-hot encodes the provided DNA sequence and returns the one-hot encoding

    Arguments:
        dna::str
            Sequence to be encoded
    
    Returns:
        onehot::np.array(int)
            List of 1's and 0's of representing the one-hot encoding
            
            
split_negatives(negative_seqs):
    Takes in a list of DNA sequences, and breaks up each sequence into a list of 17-mers. If the original sequence is shorter 
    than 17 bp it is thrown away. Returns a list of 17-mers, without duplicates

    Arguments:
        negative_seqs::[str]
            List of negative binding sequences to be split apart.
    
    Returns:
        negative_17mers::[str]
            List of all 17-mers that are subsequences of the sequences in negative_seqs. No duplicates. 
            
            
train_test_split(x, train_proportion=.9):
    Given a dataset with n observations, splits it into a training dataset of length train_proportion*n and testing dataset of length (1-train_proportion)*n.
    Note: This function only works if all the observations in the dataset are unique. I intentionally make that the case in my data pre-processing

    Arguments:
        x::[any]
            Dataset to be split
    
    Returns:
        train_set::[any]
            Training dataset consisting of train_proportion*n observations randomly sampled from the input dataset
        test_set::[any]
            Testing dataset consisting of (1-train_proportion)*n observations. Is all of the observations from x that are not in train_set.
            
            
kfold_CV(x, k=5):
    Folds an input dataset for k-fold cross-validation. Each fold will have a test dataset of length len(x)/k, and a training set of len(x) - len(test dataset).
    Each of the k test datasets generated are mutually exclusive with one another (no overlap of observations)

    Arguments:
        x::[any]
            Dataset to be folded 
    
    Returns:
        folds::[([any], [any])]
            List of tuples, each tuple being one of the k folds holding a training set and a testing set. The training set of each fold is in position 0 of its 
            tuple, and the testing set of each fold is in position 1 of its tuple
```

### Data I/O Function Documentation

```
fasta_reader(filepath):
    Reads in a fasta file located at the filepath and returns a list of the sequences in the file

    Arguments:
        filepath::str
            The path to the fasta file that is being read
    
    Returns:
        sequences::[string]
            List of strings representing the DNA sequences encoded in the fasta file
            
            
line_reader(filepath):
    Reads in a sequence file holding one sequence per line. Returns a list of these sequences
    
    Arguments:
        filepath::str
            The path to the sequence file that is being read
    
    Returns:
        sequences::[string]
            List of strings representing the DNA sequences encoded in the sequence file


write_output(outfile_path, sequences, predictions):
    Specific function for writing out the data for part 5 in the requested format (2 columns, first column is the sequence, second is the predicted value)
    
    Arguments:
        outfile_path::str
            The desired path of the output file
        sequences::[str]
            List of the sequences (in DNA string format) that have had predictions performed for them.
        predictions::[float]
            List of predicted values for the sequences in "sequences". Must be in the same order and the same length of the sequences list

    Returns:
        None
```
