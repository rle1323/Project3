import numpy as np
import random
from scripts import io, NN
from scripts import functions as f
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

# DOCUMENT DESCRIPTION: MAIN SCRIPT FOR SCRIPT TESTING AND CALLING MY IMPLEMENTED NEURAL NETWORK. PROJECT QUESTIONS ARE DONE IN ORDER ON THIS DOCUMENT, AS ANNOTATED.
# EVERYTHING IS COMMENTED OUT CURRENTLY. SECTIONS CAN BE INDIVIDUALLY UNCOMMENTED AND WILL RUN.

# set seeds for reproducibility
random.seed(1)
np.random.seed(1)

'''
############################# PART 1 #####################################
# DESCRIPTION: TRAINING AND TESTING 8X3X8 AUTOENCODER ON 8X8 IDENTITY MATRIX
## Question 1
# make 8x8 identity matrix
identity = np.identity(8)

# make the autoencoder
bit_autoencoder = NN.NeuralNetwork(architecture = [NN.DenseLayer(8,3, "sigmoid"), NN.DenseLayer(3,8, "sigmoid")], 
    seed = 5, 
    lr = .2, 
    loss_function = "mse")

# train the autoencoder on the identity matrix, and save the average loss for each epoch
losses = bit_autoencoder.fit(x = identity, y = identity, max_epochs=50000)

# get the predictions for each 8-bit vector from the trained autoencoder
predictions = []
for vec in identity:
    prediction = bit_autoencoder.predict(vec)
    predictions.append(prediction)


# plot each prediction vector as a barplot
for pred, true in zip(predictions, identity):
    x = range(1,9)
    plt.bar(x = x, height = pred)
    plt.xticks(ticks = x, labels = true)
    plt.title("Prediction for bit vector " + np.array_str(true))
    plt.xlabel("Bit vector value")
    plt.ylabel("Predicted value")
    plt.show()

# plot the loss by epoch
epochs = range(1, 50001)
plt.scatter(epochs, losses, s = 20)
plt.xlabel('Epoch')
plt.ylabel("Loss (MSE)")
plt.show()


# Part 1 complete
############################# PART 2 #####################################
#DESCRIPTION: TRAIN A CLASSIFICATION NETWORK ON THE POSITIVE AND NEGATIVE SEQUENCES AND REPORT RESULTS
#### QUESTION 3

# read in positive sequences
positive_seqs = io.line_reader("data/rap1-lieb-positives.txt")

# read in the negative seqs
negative_seqs = io.fasta_reader("data/yeast-upstream-1k-negative.fa")
# split them up so that we have a list of 17-mers. function also removes duplicates
negative_subs = f.split_negatives(negative_seqs)
for sub in negative_subs:
    if len(sub) != 17:
        print("HERE")
print("DONE")

# remove 17mers from the negative set that appear in the positive set
negative_subs = list(set(negative_subs) - set(positive_seqs))

# make a training and testing set for both positives and negatives
positive_train, positive_test = f.train_test_split(positive_seqs)
negative_train, negative_test = f.train_test_split(negative_subs)

# append the train and test lists together
x_train = [y for x in [positive_train, negative_train] for y in x]
x_test = [y for x in [positive_test, negative_test] for y in x]

# make label sets for training and testing sets
y_train = np.zeros(len(x_train))
y_train[:len(positive_train)] = 1

y_test = np.zeros(len(x_test))
y_test[:len(positive_test)] = 1

# one-hot encode training and testing sets
x_train_ohe = []
for seq in x_train:
    seq_ohe = f.dna_onehot(seq)
    x_train_ohe.append(seq_ohe)
x_train_ohe = np.asarray(x_train_ohe)

x_test_ohe = []
for seq in x_test:
    seq_ohe = f.dna_onehot(seq)
    x_test_ohe.append(seq_ohe)
x_test_ohe = np.asarray(x_test_ohe)[:500]

# make a network
EPOCHS = 100
test_nn = NN.NeuralNetwork(architecture = [NN.DenseLayer(68,8, "relu"), NN.DenseLayer(8,1, "sigmoid")],
    seed = 5, 
    lr = .2, 
    loss_function = "mse")

# train the network
losses = test_nn.fit(x = x_train_ohe, y = y_train, max_epochs = EPOCHS, mini_batch_negatives=True)

# get the predictions from the network and show them for the first 30 examples in the test set 
for i,example in enumerate(x_test_ohe[:30]):
    pred = test_nn.predict(example)
    print(x_test[i], y_test[i], pred)

# show the loss curve
epochs = range(1, EPOCHS+1)
plt.scatter(epochs, losses, s = 20)
plt.xlabel('Epoch')
plt.ylabel("Loss (MSE)")
plt.show()

############################# PART 3 #####################################
#DESCRIPTION: TESTING PERFORMANCE OF NETWORK WITH 5-FOLD CV
# set this as a static for repeated use
K = 5
#### QUESTION 5
# read in positive sequences
positive_seqs = io.line_reader("data/rap1-lieb-positives.txt")

# read in the negative seqs
negative_seqs = io.fasta_reader("data/yeast-upstream-1k-negative.fa")
# split them up so that we have a list of 17-mers. function also removes duplicates
negative_subs = f.split_negatives(negative_seqs)

# remove 17mers from the negative set that appear in the positive set
negative_subs = list(set(negative_subs) - set(positive_seqs))
# sample the negative data because the full dataset takes too long
negative_subs = random.sample(negative_subs, 500000)

# fold the data separately by class for 5-fold CV 
positive_folds = f.kfold_CV(positive_seqs, k=K)
negative_folds = f.kfold_CV(negative_subs, k=K)

# one-hot encode the positive and negative seqs, and put them into the same fold. Also make label sets for each fold
# note: it is necessary to one-hot encode after folding because my folding function hashes lists, and one-hot arrays are not hashable objects
x_folds = []
y_folds = []
for i in range(K):
    positive_train_ohe = [f.dna_onehot(seq) for seq in positive_folds[i][0]]
    positive_test_ohe = [f.dna_onehot(seq) for seq in positive_folds[i][1]]
    negative_train_ohe = [f.dna_onehot(seq) for seq in negative_folds[i][0]]
    negative_test_ohe = [f.dna_onehot(seq) for seq in negative_folds[i][1]]
    # append the train and test lists together for this fold
    x_train = [y for x in [positive_train_ohe, negative_train_ohe] for y in x]
    x_test = [y for x in [positive_test_ohe, negative_test_ohe] for y in x]
    # make label sets for training and testing sets
    y_train = np.zeros(len(x_train))
    y_train[:len(positive_train_ohe)] = 1
    y_test = np.zeros(len(x_test))
    y_test[:len(positive_test_ohe)] = 1

    # add these to the main folded datasets
    x_folds.append( (np.asarray(x_train), np.asarray(x_test)) )
    y_folds.append((y_train, y_test))

# OK, now we actually get to the neural net fitting and testing. Fit a model for each training fold and test it on the testing fold. 
# get roc for each testing set results
aucs = []
for i in range(K):
    # pull out x and y values from previously made lists
    x_train = x_folds[i][0]
    y_train = y_folds[i][0]
    x_test = x_folds[i][1]
    y_test = y_folds[i][1]

    # make a model object
    test_nn = NN.NeuralNetwork(architecture = [NN.DenseLayer(68,8, activation_function="relu"), NN.DenseLayer(8,1, activation_function="sigmoid")],
        seed = 5, 
        lr = .2, 
        loss_function = "mse")
    
    # fit the model on the training data
    losses = test_nn.fit(x = x_train, y = y_train, max_epochs = 75, mini_batch_negatives=True)

    # get predictions on the test dataset
    preds = []
    for example in x_test:
        prd = test_nn.predict(example)
        preds.append(prd)
    
    auc = roc_auc_score(y_true=y_test, y_score = preds)
    aucs.append(auc)
    print("AUC for fold", i, auc)

# print average auc across folds
print("AVERAGE AUC ACROSS FOLDS", sum(aucs)/len(aucs))


############################# PART 4 #####################################
#DESCRIPTION: TESTING IMPLEMENENTATIONS OF L2 AND DROPOUT REGULARIZATIONS WITH 5-FOLD CV
# set this as a static for repeated use
K = 5
#### QUESTION 5
# read in positive sequences
positive_seqs = io.line_reader("data/rap1-lieb-positives.txt")

# read in the negative seqs
negative_seqs = io.fasta_reader("data/yeast-upstream-1k-negative.fa")
# split them up so that we have a list of 17-mers. function also removes duplicates
negative_subs = f.split_negatives(negative_seqs)

# remove 17mers from the negative set that appear in the positive set
negative_subs = list(set(negative_subs) - set(positive_seqs))
# sample the negative data because the full dataset takes too long
negative_subs = random.sample(negative_subs, 500000)

# fold the data separately by class for 5-fold CV 
positive_folds = f.kfold_CV(positive_seqs, k=K)
negative_folds = f.kfold_CV(negative_subs, k=K)

# one-hot encode the positive and negative seqs, and put them into the same fold. Also make label sets for each fold
# note: it is necessary to one-hot encode after folding because my folding function hashes lists, and one-hot arrays are not hashable objects
x_folds = []
y_folds = []
for i in range(K):
    positive_train_ohe = [f.dna_onehot(seq) for seq in positive_folds[i][0]]
    positive_test_ohe = [f.dna_onehot(seq) for seq in positive_folds[i][1]]
    negative_train_ohe = [f.dna_onehot(seq) for seq in negative_folds[i][0]]
    negative_test_ohe = [f.dna_onehot(seq) for seq in negative_folds[i][1]]
    # append the train and test lists together for this fold
    x_train = [y for x in [positive_train_ohe, negative_train_ohe] for y in x]
    x_test = [y for x in [positive_test_ohe, negative_test_ohe] for y in x]
    # make label sets for training and testing sets
    y_train = np.zeros(len(x_train))
    y_train[:len(positive_train_ohe)] = 1
    y_test = np.zeros(len(x_test))
    y_test[:len(positive_test_ohe)] = 1

    # add these to the main folded datasets
    x_folds.append( (np.asarray(x_train), np.asarray(x_test)) )
    y_folds.append((y_train, y_test))

# OK, now we actually get to the neural net fitting and testing. Fit a model for each training fold and test it on the testing fold. 
# get roc for each testing set results

best_l2 = 0
best_dropout = 0
for l in [.05,.1,.15,.2,.25,.3,.35,.4,.45,.5]:
    print(l)
    l2_aucs = []
    dropout_aucs = []
    for i in range(K):
        # pull out x and y values from previously made lists
        x_train = x_folds[i][0]
        y_train = y_folds[i][0]
        x_test = x_folds[i][1]
        y_test = y_folds[i][1]

        # make a model object with l2 reg in first layer
        l2_nn = NN.NeuralNetwork(architecture = [NN.DenseLayer(68,8, activation_function="relu", regularization="L2", lamba = l/100), NN.DenseLayer(8,1, activation_function="sigmoid")],
            seed = 5, 
            lr = .2, 
            loss_function = "mse")
        
        # fit the model on the training data
        losses = l2_nn.fit(x = x_train, y = y_train, max_epochs = 100, mini_batch_negatives=True)

        # get predictions on the test dataset
        preds = []
        for example in x_test:
            prd = l2_nn.predict(example)
            preds.append(prd)

        # generate auc scores and save them 
        auc = roc_auc_score(y_true=y_test, y_score = preds)
        l2_aucs.append(auc)

        ###### DO EVERYTHING THE SAME FOR DROPOUT
        # make a model object with l2 reg in first layer
        dropout_nn = NN.NeuralNetwork(architecture = [NN.DenseLayer(68,8, activation_function="relu", regularization="Dropout", lamba = l), NN.DenseLayer(8,1, activation_function="sigmoid")],
            seed = 5, 
            lr = .2, 
            loss_function = "mse")
        
        # fit the model on the training data
        losses = dropout_nn.fit(x = x_train, y = y_train, max_epochs = 100, mini_batch_negatives=True)

        # get predictions on the test dataset
        preds = []
        for example in x_test:
            prd = dropout_nn.predict(example)
            preds.append(prd)
        
        # generate auc scores and save them 
        auc = roc_auc_score(y_true=y_test, y_score = preds)
        dropout_aucs.append(auc)

    l2_auc = sum(l2_aucs)/len(l2_aucs)
    dropout_auc = sum(dropout_aucs)/len(dropout_aucs)
    print("L2", l2_auc)
    print("Dropout", dropout_auc)
    if l2_auc > best_l2: 
        print("NEW BEST l2")
        best_l2 = l2_auc
        l2_output = (l/100, best_l2)
    if dropout_auc > best_dropout: 
        print("NEW BEST DROPOUT")
        best_dropout = dropout_auc
        dropout_output = (l, best_dropout)

print("BEST PERFORMING DROPOUT", dropout_output)
print("BEST PERFORMING L2", l2_output)


############################# PART 5 #####################################
### DESCRIPTION: DOING FULL TRAINING OF NETWORK AND GENERATING PREDICTIONS FOR TEST DATASET
# read in positive sequences
positive_seqs = io.line_reader("data/rap1-lieb-positives.txt")

# read in the negative seqs
negative_seqs = io.fasta_reader("data/yeast-upstream-1k-negative.fa")
# split them up so that we have a list of 17-mers. function also removes duplicates
negative_subs = f.split_negatives(negative_seqs)

# remove 17mers from the negative set that appear in the positive set
negative_subs = list(set(negative_subs) - set(positive_seqs))

# no need to subset the data because we arent doing 5-fold CV 

# append the train and test lists together
x = [y for x_train in [positive_seqs, negative_subs] for y in x_train]

# make label sets for training and testing sets
y = np.zeros(len(x))
y[:len(positive_seqs)] = 1

# one-hot encode training  sets
x_ohe = [f.dna_onehot(seq) for seq in x]
x_ohe = np.asarray(x_ohe)

# define the network
test_nn = NN.NeuralNetwork(architecture = [NN.DenseLayer(68,8, "relu"), NN.DenseLayer(8,1, "sigmoid")],
    seed = 5, 
    lr = .2, 
    loss_function = "mse")

# train the network
losses = test_nn.fit(x_ohe, y, max_epochs=500, mini_batch_negatives=True)

# read in test dataset
test_seqs = io.line_reader("data/rap1-lieb-test.txt")
# one-hot encode the test dataset
test_ohe = [f.dna_onehot(seq) for seq in test_seqs]
test_ohe = np.asarray(test_ohe)

predictions = []
for i, seq in enumerate(test_ohe):
    pred = test_nn.predict(seq)
    print(test_seqs[i], pred)
    predictions.append(pred)

# write the output 
io.write_output("data/output.txt", test_seqs, predictions)
'''


