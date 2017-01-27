import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

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
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)


  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  pass
  #kenneth yu
  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]

  for i in xrange(num_train):
    scores = X[i].dot(W)      # scores shape (1,C)
    #shift scores
    scores -= np.max(scores)
    sum_p = np.sum(np.exp(scores),axis=-1)

    Li = -scores[y[i]] + np.log(sum_p)
    loss += Li

    #calc gradient.
    #  dW[:,j] += np.transpose(X[i,:])
    #  dW[:,y[i]] -= np.transpose(X[i,:])


    for j in xrange(num_classes):
      # dLi / dWj
      dW[:, j ] +=  (np.exp(scores[j]) / sum_p ) * X[i,:]
    #end of for j

    dW[:,y[i]] -= X[i,:]
    #end of for i

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  #. average gradient.
  dW /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)  #elementwise product. not matrix dot.
  # kenneth yu. add the reg gradient.
  #dW +=  0.5 * 2 * reg * W
  dW += reg * W


  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  pass
  #kenneth yu.
  # compute the loss and the gradient
  #num_classes = W.shape[1]
  num_train = X.shape[0]

  scores = X.dot(W)      # scores shape (N,C)
  #shift scores by max of each row(each sample scores). reshape to avoid ambiguity of broadcasting
  #in case N=C ....
  scores -= np.max(scores, axis=-1).reshape((scores.shape[0], -1))

  #sum_p each row corresponds to each shifted sample scores.
  sum_p = np.sum(np.exp(scores),axis=-1)   #sum_p shape(num_train, )

  loss = - np.sum(scores[np.array(xrange(scores.shape[0])), y]) + np.sum(np.log(sum_p))

  #calc gradient.
  # dLi / dWj
  # dW[:, j ] +=  (np.exp(scores[j]) / sum_p ) * X[i,:]
  # dW[:,y[i]] -= X[i,:]

  # mark the y[i]
  Indicator = np.zeros_like(scores)       #shape (N,C)
  Indicator[np.array(xrange(Indicator.shape[0])), y] += -1

  # mark the derivative of logs
  d_log = np.divide(np.exp(scores), sum_p.reshape(sum_p.shape[0],-1))        #shape (N,C)

  dW = np.transpose(X).dot(Indicator + d_log)  #shape (D,C)

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  #. average gradient.
  dW /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)  #elementwise product. not matrix dot.
  # kenneth yu. add the reg gradient.
  #dW +=  0.5 * 2 * reg * W
  dW += (reg * W)


  #dW = (np.transpose(X)).dot(indicator)

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

