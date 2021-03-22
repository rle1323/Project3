from scripts import NN
from scripts import io 
from scripts import functions as f
import numpy as np

########################## Testing utility functions in functions.py
def test_sigmoid():
    # true values calculated from http://www.tinkershop.net/ml/sigmoid_calculator.html
    true_y = np.asarray([.5, .27, .73])
    test_x = np.asarray([0, -1, 1])
    # round answers to two decimal places so they can be compared
    test_y = np.round(f.sigmoid(test_x), decimals=2)
    # check equivalence
    assert np.array_equal(true_y, test_y)

def test_sigmoid_prime():
    # true values calculated from http://www.tinkershop.net/ml/sigmoid_calculator.html
    true_y = np.asarray([.25, .20, .20])
    test_x = np.asarray([0, -1, 1])
    # round answers to two decimal places so they can be compared
    test_y = np.round(f.sigmoid_prime(test_x), decimals=2)
    # check equivalence
    assert np.array_equal(true_y, test_y)

def test_relu():
    test_x = np.asarray([-1,0,1,5])
    test_y = f.relu(test_x)
    true_y = np.asarray([0,0,1,5])
    assert np.array_equal(true_y, test_y)

def test_relu_prime():
    test_x = np.asarray([-1,0,1,5])
    test_y = f.relu_prime(test_x)
    true_y = np.asarray([0,0,1,1])
    assert np.array_equal(true_y, test_y)
    
def test_mse():
    test_y = np.asarray([1, .5, 1, 0])
    test_yhat = np.asarray([.5, .5, .5, .5])
    test_mse = f.mse(test_y, test_yhat)
    assert test_mse == (.75/4)

def test_mse_prime():
    test_y = np.asarray([1, .5, 1, 0])
    test_yhat = np.asarray([.5, .5, .5, .5])
    test_mse = f.mse_prime(test_y, test_yhat)
    assert np.array_equal(test_mse, (2*(test_yhat-test_y))/test_y.size)

def test_bce():
    test_y = np.asarray([1, .5, 1, 0])
    test_yhat = np.asarray([.5, .5, .5, .5])
    test_bce = f.binary_cross_entropy(test_y, test_yhat)
    tf_bce = -(test_y)*np.log(test_yhat) - (1-test_y)*np.log(1 - test_yhat)
    assert np.array_equal(test_bce,tf_bce)

def test_onehot():
    test_oh = f.dna_onehot("ATCG")
    test_answer = [1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1]
    assert all(test_oh == test_answer)

def test_split_negatives():
    test_split = f.split_negatives(["AAAACCCCTTTTGGGGACGT"])
    test_answer = ['AAAACCCCTTTTGGGGA', 'AACCCCTTTTGGGGACG', 'ACCCCTTTTGGGGACGT', 'AAACCCCTTTTGGGGAC']
    for seq in test_split:
        assert seq in test_answer

def test_train_test_split():
    test_x = [1, 2, 4, 3]
    x_train, x_test = f.train_test_split(test_x, train_proportion = .75)
    assert len(x_train) == 3
    assert len(x_test) == 1

def test_kfold_CV():
    test_x = [1, 2, 4, 3, 5, 7, 8,6]
    folds = f.kfold_CV(test_x, k=4)
    for fold in folds:
        assert len(fold[0]) == 6
        assert len(fold[1]) == 2


## TESTING DATA IO
def test_fasta_reader():
    seqs = io.fasta_reader("data/yeast-upstream-1k-negative.fa")
    assert len(seqs) == 3164

def test_line_reader():
    seqs = io.line_reader("data/rap1-lieb-positives.txt")
    assert len(seqs) == 137


def test_encoder():
    assert True

def test_encoder_relu():
    assert True

def test_one_d_ouput():
    assert False
