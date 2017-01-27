import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        #kenneth yu. calc gradient.
        dW[:,j] += np.transpose(X[i,:])
        dW[:,y[i]] -= np.transpose(X[i,:])

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  #kenneth yu. average gradient.
  dW /= num_train



  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)  #elementwise product. not matrix dot.
  # kenneth yu. add the reg gradient.
  dW +=  0.5 * 2 * reg * W

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################
  #kenneth yu.

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  pass

  #by kenneth yu.

  # compute the loss and the gradient
  # num_classes = W.shape[1]
  num_train = X.shape[0]
  # loss = 0.0

  scores = X.dot(W)                # (N , C) matrix
  #correct_class_score = scores[y[i]]
  #each class score subtract the label score for each data.
  #scores[np.array(xrange(num_train)), y] shape (num_train, )
  #print 'scores[np.array(xrange(num_train)), y].shape -->', scores[np.array(xrange(num_train)), y].shape

  #broadcast array math. note: margin[i][y[i]] = delta now.
  margin = scores - scores[np.array(xrange(num_train)), y].reshape((scores.shape[0], -1))  + 1  # delta = 1
  #make margin[i][y[i]] = 0.
  #margin[np.array(xrange(num_train)), y] *= 0
  margin[np.array(xrange(margin.shape[0])), y] *= 0
  #use boolean index as Max(, ) function.
  loss = np.sum(margin[margin > 0])

  #calc gradient. indicators need margin boolean index.
  #for each margin > 0 :
  # dW[:,j] += np.transpose(X[i,:])
  # dW[:,y[i]] -= np.transpose(X[i,:])
  indicator = np.int8(margin > 0)
  indicator[np.array(xrange(indicator.shape[0])), y]  = -np.sum(indicator, axis = -1)

  #print 'indicator shape:', indicator.shape   #should be (N,10)
  #print 'indicator :', indicator

  dW = (np.transpose(X)).dot(indicator)


  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  #kenneth yu. average gradient.
  dW /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)  #elementwise product. not matrix dot.
  # kenneth yu. add the reg gradient.
  #dW +=  (0.5 * 2) * reg * W
  dW += (reg * W)

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return loss, dW

