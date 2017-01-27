# Run some setup code for this notebook.

import random

import numpy as np
from cs231n.data_utils import load_CIFAR10
import matplotlib.pyplot as plt



# This is a bit of magic to make matplotlib figures appear inline in the notebook
# rather than in a new window.
# %matplotlib inline

"""
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
"""

"""
# Some more magic so that the notebook will reload external python modules;
# see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython
# %reload_ext autoreload
# %autoreload 2
"""


# Load the raw CIFAR-10 data.
cifar10_dir = 'cs231n/datasets/cifar-10-batches-py'
X_origin_train, y_origin_train, X_origin_test, y_origin_test = load_CIFAR10(cifar10_dir)

# As a sanity check, we print out the size of the training and test data.
print 'Training data shape: ', X_origin_train.shape
print 'Training labels shape: ', y_origin_train.shape
print 'Test data shape: ', X_origin_test.shape
print 'Test labels shape: ', y_origin_test.shape


#comment to accelerate. kennethyu.
# Visualize some examples from the dataset.
# We show a few examples of training images from each class.
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
num_classes = len(classes)
samples_per_class = 7
for y, cls in enumerate(classes):
    idxs = np.flatnonzero(y_train == y)
    idxs = np.random.choice(idxs, samples_per_class, replace=False)
    for i, idx in enumerate(idxs):
        plt_idx = i * num_classes + y + 1
        plt.subplot(samples_per_class, num_classes, plt_idx)
        plt.imshow(X_train[idx].astype('uint8'))
        plt.axis('off')
        if i == 0:
            plt.title(cls)
plt.show()

# Subsample the data for more efficient code execution in this exercise
num_training = 5000

mask = range(num_training)
X_train = X_origin_train[mask]
y_train = y_origin_train[mask]

del X_origin_train, y_origin_train

num_test = 500

mask = range(num_test)
X_test = X_origin_test[mask]
y_test = y_origin_test[mask]

del X_origin_test , y_origin_test

print  'subsample of X_train.shape', X_train.shape
print  'subsample of X_test.shape', X_test.shape

X_test_bak = np.copy(X_test)

# Reshape the image data into rows
X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))
print X_train.shape, X_test.shape


from cs231n.classifiers import KNearestNeighbor


# Create a kNN classifier instance.
# Remember that training a kNN classifier is a noop:
# the Classifier simply remembers the data and does no further processing
classifier = KNearestNeighbor()
classifier.train(X_train, y_train)

####
#We would now like to classify the test data with the kNN classifier. Recall that we can break down this
#  process into two steps:
#First we must compute the distances between all test examples and all train examples.
#Given these distances, for each test example we find the k nearest examples and have them vote for the
# label .Lets begin with computing the distance matrix between all training and test examples.
# For example, if there are Ntr training examples and Nte test examples, this stage should result
#  in a Nte x Ntr matrix where each element (i,j) is the distance between the i-th test and j-th train
# example.
# First, open cs231n/classifiers/k_nearest_neighbor.py and implement the function
#  compute_distances_two_loops that uses a (very inefficient) double loop over all pairs of (test, train)
# examples and computes the distance matrix one element at a time.
###


# Open cs231n/classifiers/k_nearest_neighbor.py and implement
# compute_distances_two_loops.

# Test your implementation:
dists = classifier.compute_distances_two_loops(X_test)
print 'dists array shape: ', dists.shape


# We can visualize the distance matrix: each row is a single test example and
# its distances to training examples
plt.imshow(dists, interpolation='none')
plt.show()

"""
Inline Question #1: Notice the structured patterns in the distance matrix, where some rows or columns
are visible brighter. (Note that with the default color scheme black indicates low distances while
white indicates high distances.)
What in the data is the cause behind the distinctly bright rows?What causes the columns?

Your Answer: fill this in.
kenneth yu: the bright row means the corresponding test example which keeps the large distance gap between
            each of the training examples, maybe this test example is a pure dark color(e.g., night picture) image.
            samely, the bright column means the corresponding training example is a pure dark color image.
"""

# show the test image with highest distance related to all test examples.
dist_sum_of_te = np.sum(dists,axis=1)
idx_max_dist_te = np.argmax(dist_sum_of_te,axis=0)
#te_image = np.reshape(X_test[idx_max_dist_te],(32,32,3))
plt.imshow(X_test_bak[idx_max_dist_te])
plt.show()


# Now implement the function predict_labels and run the code below:
# We use k = 1 (which is Nearest Neighbor).
y_test_pred = classifier.predict_labels(dists, k=1)

# Compute and print the fraction of correctly predicted examples
num_correct = np.sum(y_test_pred == y_test)
accuracy = float(num_correct) / num_test
print 'k--1, Got %d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy)


#You should expect to see approximately 27% accuracy. Now lets try out a larger k, say k = 5:

y_test_pred = classifier.predict_labels(dists, k=5)
num_correct = np.sum(y_test_pred == y_test)
accuracy = float(num_correct) / num_test
print 'k -- 5, Got %d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy)

#You should expect to see a slightly better performance than with `k = 1`.


# Now lets speed up distance matrix computation by using partial vectorization
# with one loop. Implement the function compute_distances_one_loop and run the
# code below:
dists_one = classifier.compute_distances_one_loop(X_test)


# To ensure that our vectorized implementation is correct, we make sure that it
# agrees with the naive implementation. There are many ways to decide whether
# two matrices are similar; one of the simplest is the Frobenius norm. In case
# you haven't seen it before, the Frobenius norm of two matrices is the square
# root of the squared sum of differences of all elements; in other words, reshape
# the matrices into vectors and compute the Euclidean distance between them.
difference = np.linalg.norm(dists - dists_one, ord='fro')
print 'Difference was: %f' % (difference, )
if difference < 0.001:
  print 'Good! The distance matrices are the same'
else:
  print 'Uh-oh! The distance matrices are different'


# Now implement the fully vectorized version inside compute_distances_no_loops
# and run the code
dists_two = classifier.compute_distances_no_loops(X_test)

# check that the distance matrix agrees with the one we computed before:
difference = np.linalg.norm(dists - dists_two, ord='fro')
print 'Difference was: %f' % (difference, )
if difference < 0.001:
  print 'Good! The distance matrices are the same'
else:
  print 'Uh-oh! The distance matrices are different'


# Let's compare how fast the implementations are
def time_function(f, *args):
  # Call a function f with args and return the time (in seconds) that it took to execute.
  import time
  tic = time.time()
  f(*args)
  toc = time.time()
  return toc - tic

two_loop_time = time_function(classifier.compute_distances_two_loops, X_test)
print 'Two loop version took %f seconds' % two_loop_time

one_loop_time = time_function(classifier.compute_distances_one_loop, X_test)
print 'One loop version took %f seconds' % one_loop_time

no_loop_time = time_function(classifier.compute_distances_no_loops, X_test)
print 'No loop version took %f seconds' % no_loop_time

# you should see significantly faster performance with the fully vectorized implementation

#Cross-validation
#We have implemented the k-Nearest Neighbor classifier but we set the value k = 5 arbitrarily.
#We will now determine the best value of this hyperparameter with cross-validation.

num_folds = 5
k_choices = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100]

X_train_folds = []
y_train_folds = []
################################################################################
# TODO:                                                                        #
# Split up the training data into folds. After splitting, X_train_folds and    #
# y_train_folds should each be lists of length num_folds, where                #
# y_train_folds[i] is the label vector for the points in X_train_folds[i].     #
# Hint: Look up the numpy array_split function.                                #
################################################################################
pass
#kenneth yu
X_train_folds = np.array_split(X_train, num_folds)
y_train_folds = np.array_split(y_train, num_folds)


################################################################################
#                                 END OF YOUR CODE                             #
################################################################################

# A dictionary holding the accuracies for different values of k that we find
# when running cross-validation. After running cross-validation,
# k_to_accuracies[k] should be a list of length num_folds giving the different
# accuracy values that we found when using that value of k.

## kenneth yu.

k_to_accuracies = {}
for k in k_choices:
  k_to_accuracies[k] = []


################################################################################
# TODO:                                                                        #
# Perform k-fold cross validation to find the best value of k. For each        #
# possible value of k, run the k-nearest-neighbor algorithm num_folds times,   #
# where in each case you use all but one of the folds as training data and the #
# last fold as a validation set. Store the accuracies for all fold and all     #
# values of k in the k_to_accuracies dictionary.                               #
################################################################################
pass
#kenneth yu
#for i in xrange(num_folds):
for i in xrange(len(X_train_folds)):
  # validation using the ith data fold, using different K and save the results.
  X_cross_validation = X_train_folds[i]
  y_cross_validation = y_train_folds[i]

  # train using the other (num_folds - 1)  datafolds.
  # trick:if i == num_folds, then X_train_folds[i+1:] will return none.
  #X_cross_train = np.array((X_train_folds[0:i]).append(X_train_folds[(i+1):]))
  #y_cross_train = np.array((y_train_folds[0:i]).append(y_train_folds[(i+1):]))
  X_cross_train = [data for (idx,data) in enumerate(X_train_folds) if idx != i]
  #for idx,data in enumerate(X_train_folds):
  #  if idx != i:
      #ate((X_cross_train,data), axis = 0)
  #   X_cross_train.append(data)
  X_cross_train = np.reshape(X_cross_train,(len(X_cross_train) * (X_cross_train[0].shape[0]), -1),)

  y_cross_train = []

  y_cross_train = [data for (idx, data) in enumerate(y_train_folds) if idx != i]
  #for idx,data in enumerate(y_train_folds):
  #  if idx != i:
      #np.concatenate((y_cross_train,data), axis = 0)
  #    y_cross_train.append(data)
  y_cross_train = np.reshape(y_cross_train,(len(y_cross_train) * (y_cross_train[0].shape[0]), -1),)

  #reshape X to ( num_cross_validation, 3072) , y to (num_cross_validation, 1) array.
  #X_cross_train = np.reshape(X_cross_train, (-1,X_cross_train[0].shape[-1]))
  #y_cross_train = np.reshape(y_cross_train, (-1,y_cross_train[0].shape[-1]))
  #train.
  classifier.train(X_cross_train, y_cross_train)

  #validation on each k. use one loop solution.
  for k in k_choices:
    y_cross_validation_pred = classifier.predict(X_cross_validation, k, num_loops=1)
    num_correct = np.sum(y_cross_validation_pred == y_cross_validation)
    accuracy = float(num_correct) / y_cross_validation_pred.shape[0]
    print 'fold - %d K -- %d Got %d / %d correct => accuracy: %f' % (i, k, num_correct, y_cross_validation_pred.shape[0],
                                                                     accuracy)

    #save the accuracy for each K using list of length num_folds.
    k_to_accuracies[k].append(accuracy)

################################################################################
#                                 END OF YOUR CODE                             #
################################################################################

# Print out the computed accuracies
for k in sorted(k_to_accuracies):
    for accuracy in k_to_accuracies[k]:
        print 'k = %d, accuracy = %f' % (k, accuracy)


# plot the raw observations
for k in k_choices:
  accuracies = k_to_accuracies[k]
  plt.scatter([k] * len(accuracies), accuracies)

# plot the trend line with error bars that correspond to standard deviation
accuracies_mean = np.array([np.mean(v) for k,v in sorted(k_to_accuracies.items())])
accuracies_std = np.array([np.std(v) for k,v in sorted(k_to_accuracies.items())])
plt.errorbar(k_choices, accuracies_mean, yerr=accuracies_std)
plt.title('Cross-validation on k')
plt.xlabel('k')
plt.ylabel('Cross-validation accuracy')
plt.show()



# Based on the cross-validation results above, choose the best value for k,
# retrain the classifier using all the training data, and test it on the test
# data. You should be able to get above 28% accuracy on the test data.
#kenneth yu . the best K is 1.woops.
best_k = 10

classifier = KNearestNeighbor()
classifier.train(X_train, y_train)
#y_test_pred = classifier.predict(X_test, k=best_k)
#kenneth yu. use 2 loops. smaller memory.
y_test_pred = classifier.predict(X_test, k=best_k, num_loops=2)

# Compute and display the accuracy
num_correct = np.sum(y_test_pred == y_test)
accuracy = float(num_correct) / num_test
print 'k --- %d, Got %d / %d correct => accuracy: %f' % (best_k, num_correct, num_test, accuracy)


############Finnaly , kenneth yu got 29% accuracy on K=1 ,1000 training data and 100 test data. Sep9.#######################


