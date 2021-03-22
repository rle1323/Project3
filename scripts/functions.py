import numpy as np
import random, math

###################### ACTIVATION FUNCTIONS (with derivatives)######################
def sigmoid(x):
    """
    Expression that calculates the sigmoid of a number x
    This is used as a non-linear activation function.

    Arguments:
        x::float or np.array(float)
            Scalar or array of scalars that expression is performed on
    
    Returns::
        y::float or np.array(float)
            Result of sigmoid(x)
    """
    y = 1/(1+np.exp(-x))
    return y


def sigmoid_prime(x):
    """
    Expression that calculates the derivative function of sigmoid of a number x
    The integral of this function is used as a non-linear activation function, and this derivative function is called in backpropogation
    
    Arguments:
        x::float or np.array(float)
            Scalar or array of scalars that expression is performed on
    
    Returns::
        y::float or np.array(float)
            Result of sigmoid(x)
    """
    y = sigmoid(x) * (1-sigmoid(x))
    return y


def relu(x):
    """
    Expression that calculates the ReLU function of a scalar. ReLU(x) = x if x > 0, and 0 if x <= 0.
    This is used as a non-linear activation function.
    
    Arguments:
        x::float or np.array(float)
            Scalar or array of scalars that expression is performed on
    
    Returns::
        y::float or np.array(float)
            Result of relu(x)
    """
    y = x * (x > 0)
    return y


def relu_prime(x):
    """
    Expression that calculates the derivative of the ReLU function of a scalar.
    The integral of this function is used as a non-linear activation function, and this derivative function is called in backpropogation
    
    Arguments:
        x::float or np.array(float)
            Scalar or array of scalars that expression is performed on
    
    Returns::
        y::float or np.array(float)
            Result of relu(x)
    """

    y = (x > 0)
    return y


###################### LOSS FUNCTIONS (with derivatives)######################
def mse(y, y_hat):
    """
    Calculates the mean-squared error between a prediction and true value

    Arguments:
        y::np.array(float)
            Ground truth labels
        y_hat::np.array(float) 
            Predicted labels
    
    Returns:
        mse::float
            Mean-squared error calculated
    """
    mse = (np.square(y - y_hat)).mean()
    return mse

def mse_prime(y, y_hat):
    """
    Expression that calculates the derivative function of mse of a prediction, given the predicted and true values
    
    Arguments:
        y::np.array(float)
            Ground truth labels
        y_hat::np.array(float) 
            Predicted labels
    
    Returns:
        mse_prime::float
            Mean-squared prime calculated
    """
    mse_prime = (2*(y_hat-y))/y.size
    return mse_prime


def binary_cross_entropy(y, y_hat):
    """
    Calculates the mean-squared error between a prediction and true value. This function is used as a neural network loss function

    Arguments:
        y::np.array(float)
            Ground truth labels
        y_hat::np.array(float) 
            Predicted labels
    
    Returns:
        bce::float
            Binary cross-entropy calculated
    """
    bce = -(y)*np.log(y_hat) - (1-y)*np.log(1 - y_hat)
    return bce


def binary_cross_entropy_prime(y, y_hat):
    """
    Expression that calculates the derivative function of binary cross-entropy of a prediction, given the predicted and true values. 
    
    Arguments:
        y::np.array(float)
            Ground truth labels
        y_hat::np.array(float) 
            Predicted labels
    
    Returns:
        bce_prime::float
            Binary cross-entropy prime calculated
    """
    if y == 1:
        bce_prime = -1/y_hat
    else:
        bce_prime = 1/(1 - y_hat)
    return bce_prime


###################### MISCELLANEOUS FUNCTIONS ######################
def dna_onehot(dna):
    """
    One-hot encodes the provided DNA sequence and returns the one-hot encoding

    Arguments:
        dna::str
            Sequence to be encoded
    
    Returns:
        onehot::np.array(int)
            List of 1's and 0's of representing the one-hot encoding
    """
    onehot = []
    for base in dna:
        if base == "A":
            onehot.append(1)
            onehot.append(0)
            onehot.append(0)
            onehot.append(0)
        elif base == "T":
            onehot.append(0)
            onehot.append(1)
            onehot.append(0)
            onehot.append(0)
        elif base == "C":
            onehot.append(0)
            onehot.append(0)
            onehot.append(1)
            onehot.append(0)
        elif base == "G":
            onehot.append(0)
            onehot.append(0)
            onehot.append(0)
            onehot.append(1)
        else:
            print("Unrecognized base", base, "in DNA sequence")
    
    return np.array(onehot)


def split_negatives(negative_seqs):
    """
    Takes in a list of DNA sequences, and breaks up each sequence into a list of 17-mers. If the original sequence is shorter 
    than 17 bp it is thrown away. Returns a list of 17-mers, without duplicates

    Arguments:
        negative_seqs::[str]
            List of negative binding sequences to be split apart.
    
    Returns:
        negative_17mers::[str]
            List of all 17-mers that are subsequences of the sequences in negative_seqs. No duplicates. 
    """
    # get the 17mers
    negative_17mers = []
    for seq in negative_seqs:
        if len(seq) >= 17:
            for i in range(len(seq) - 17 + 1):
                subseq = seq[i:i+17]
                if len(subseq) == 17:
                    negative_17mers.append(subseq)
    
    # remove duplicates
    negative_17mers = list(set(negative_17mers))

    return negative_17mers


def train_test_split(x, train_proportion=.9):
    """
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
    """
    # get the number of cases that will be used in training
    num_train = round(len(x)*train_proportion)

    # generate a random list of indices that will be the training set
    train_idx = np.random.choice(len(x), size=num_train, replace = False)
    train_set = [x[i] for i in train_idx]

    # generate a test set by removing train_set from x
    test_set = list(set(x) - set(train_set))

    return train_set, test_set


def kfold_CV(x, k=5):
    """
    Folds an input dataset for k-fold cross-validation. Each fold will have a test dataset of length len(x)/k, and a training set of len(x) - len(test dataset).
    Each of the k test datasets generated are mutually exclusive with one another (no overlap of observations)

    Arguments:
        x::[any]
            Dataset to be folded 
    
    Returns:
        folds::[([any], [any])]
            List of tuples, each tuple being one of the k folds holding a training set and a testing set. The training set of each fold is in position 0 of its tuple,
            and the testing set of each fold is in position 1 of its tuple
    """

    # assemble the testing sets by taking a random sample of the data
    testing_sets = []
    x_copy = list(x)
    for i in range(k):
        # lil shortcut if we are on the last fold
        if i == (k-1): 
            new_fold = set(x_copy)
        else:
            fold_size = math.floor(len(x_copy)/(k-i))
            # use sets because speed
            new_fold = set()
            while len(new_fold) < fold_size:
                index = random.randrange(len(x_copy))
                # add the indexed observation to the fold and remove it from the available observations for future folds
                new_fold.add(x_copy.pop(index))       
        testing_sets.append(new_fold)
    
    # assemble the proper training set for each corresponding test set
    training_sets = []
    for test_set in testing_sets:
        train_set = [seq for seq in x if seq not in test_set]
        training_sets.append(train_set)
    
    folds = [(train, list(test)) for train, test in zip(training_sets, testing_sets)]

    return folds
