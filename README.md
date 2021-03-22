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

### Neural Network Documentation

A neural network can be initialized with a NeuralNetwork() class object, and it's architecture is defined as a list of DenseLayer objects. Further documentation for both of these classes and their methods are below. A NeuralNetwork object can be trained with the fit() method, and predictions can be made with the predict() method.

#### class NeuralNetwork:
This is a class that performs all of the functions of a basic neural network. A neural network is simply a series of layers that transform a numeric input into 
an output of certain shape. These transformations can be learned to optimize a loss function, and can therefore be used to learn complex, non-linear functions. 
A lot of the methods in this particular class are simply wrappers of methods of individual layers, from the DenseLayer class. I designed it this way intentionally to break the neural network and its methods into modular components that can be easily understood on their own. 

##### NeuralNetwork Class Methods

__init__(self, architecture=[DenseLayer(68,25,f.sigmoid), DenseLayer(25,1,f.sigmoid)],lr=.05,seed=1,loss_function = "mse"):
        Initializes a NeuralNetwork object.
        
        Arguments:
                architecture::[DenseLayer()]
                        List of DenseLayer objects that together make up the design of the network. Each layer has an input shape, layer size, and activation 
                        function assigned to it. Regularization can be optionally added on a layer by layer basis
                lr::float
                        Learning rate, which determines how much weights are adjusted during backpropogation. This param should be tuned
                seed::int
                        Seed for random function seeding
                loss_function::str
                        Loss function that the network optimizes with respect to. The available options in my implementation are mean-squared error, passed as 
                        "mse", or binary cross-entrpy, passed as "bce". 

        Returns:
                None
```

class DenseLayer:
    This is a class that represents a fully connected layer in a neural network. A list of these layers defines an architecture for a "NeuralNetwork" object 
    defined later in this script.
    
    method __init__(self, input_size, layer_size, activation_function="sigmoid", regularization="None", lamba=.2):
        Initializes a DenseLayer object.

        Arguments:
            input_size::int
                Defines the number of incoming weights for each node in this layer. This will also be equal to the number of nodes in the previous layer,
                or the size of the input vector if this is the input layer. 
            layer_size::int
                Number of nodes in this layer. 
            activation_function::str (optional)
                String defining the non-linear activation function to be used in the nodes of this layer. The options in this implemenation are "sigmoid" and 
                "relu". If not supplied, defaults to "sigmoid"
            regularization::str (optional)
                String defining the regularization function to be used on the outgoing weights of this layer. The options in this implementation are "None", "L2",   
                and "Dropout".
                If not supplied, defaults to "None"
            lambda::float (optional)
                Regularization parameter that weights the regularization if using L2, or defines the dropout rate if using dropout. If not supplied, defaults to .2
        Returns:
            None


```
