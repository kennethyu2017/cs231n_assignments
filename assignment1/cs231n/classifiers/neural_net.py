import numpy as np
import matplotlib.pyplot as plt


class TwoLayerNet(object):
  """
  A two-layer fully-connected neural network. The net has an input dimension of
  N, a hidden layer dimension of H, and performs classification over C classes.
  We train the network with a softmax loss function and L2 regularization on the
  weight matrices. The network uses a ReLU nonlinearity after the first fully
  connected layer.

  In other words, the network has the following architecture:

  input - fully connected layer - ReLU - fully connected layer - softmax

  The outputs of the second fully-connected layer are the scores for each class.
  """

  def __init__(self, input_size, hidden_size, output_size, std=1e-4):
    """
    Initialize the model. Weights are initialized to small random values and
    biases are initialized to zero. Weights and biases are stored in the
    variable self.params, which is a dictionary with the following keys:

    W1: First layer weights; has shape (D, H)
    b1: First layer biases; has shape (H,)
    W2: Second layer weights; has shape (H, C)
    b2: Second layer biases; has shape (C,)

    Inputs:
    - input_size: The dimension D of the input data.
    - hidden_size: The number of neurons H in the hidden layer.
    - output_size: The number of classes C.
    """
    self.params = {}
    self.params['W1'] = std * np.random.randn(input_size, hidden_size)
    self.params['b1'] = np.zeros(hidden_size)
    self.params['W2'] = std * np.random.randn(hidden_size, output_size)
    self.params['b2'] = np.zeros(output_size)

  def loss(self, X, y=None, reg=0.0):
    """
    Compute the loss and gradients for a two layer fully connected neural
    network.

    Inputs:
    - X: Input data of shape (N, D). Each X[i] is a training sample.
    - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
      an integer in the range 0 <= y[i] < C. This parameter is optional; if it
      is not passed then we only return scores, and if it is passed then we
      instead return the loss and gradients.
    - reg: Regularization strength.

    Returns:
    If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
    the score for class c on input X[i].

    If y is not None, instead return a tuple of:
    - loss: Loss (data loss1 and regularization loss) for this batch of training
      samples.
    - grads: Dictionary mapping parameter names to gradients of those parameters
      with respect to the loss function; has the same keys as self.params.
    """
    # Unpack variables from the params dictionary
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    N, D = X.shape

    # Compute the forward pass
    scores = None
    #############################################################################
    # TODO: Perform the forward pass, computing the class scores for the input. #
    # Store the result in the scores variable, which should be an array of      #
    # shape (N, C).                                                             #
    #############################################################################
    pass
    #by kenneth yu
    #insert bias dimension into input data.
    #X reshape to (N, D+1). last column all 1. that is , add one new feature(always=1) to
    # each data.
    #X = np.hstack((X, np.ones((N, 1))))

    #insert bias row into  W1, W2 .
    #W1 reshape to (D + 1, H). W2 reshape to (H + 1,C)
    #W1 = np.vstack((W1, b1))
    #W2 = np.vstack((W2, b2))

    #print 'X insert bias shape:', X.shape        #should be (N, D+1)
    #print 'W1 insert bias reshape:', W1.shape    #should be (D+1, H)
    #print 'W2 insert bias reshape:', W2.shape    #should be (H+1,C)

    #pre-process input data
    pass

    #define activation function. ReLU
    activation_f = lambda x: (x > 0).astype('uint8') * x

    #calc scores, record the intermediate data for BP.
    #H1 layer linear SVM
    #H1_linear_score = X.dot(W1)      #shape (N,H)
    #print 'H1_linear_score shape:', H1_linear_score.shape    #should be (N,H)
    #H-1 layer output.
    h1 = activation_f(X.dot(W1) + b1)  #shape (N,H)

    #ReLU on H1 linear SVM output
    #H1_ReLU_score =  (H1_linear_score > 0).astype('uint8') * H1_linear_score   #shape (N,H)
    #print 'H1_ReLU_score shape:', H1_ReLU_score.shape    #should be (N,H)

    #insert bias dimension into  H1_ReLU_score .
    #H1_ReLU_score reshape to (N, H+1). last column all 1. that is , add one new feature(always=1) to
    # each intermediate H1 output.
    #H1_ReLU_score = np.hstack((H1_ReLU_score, np.ones((N, 1))))
    #print 'H1_ReLU_score reshape:', H1_ReLU_score.shape    #should be (N,H+1)

    #H2 layer linear SVM.
    #W2 shape (H+1,C).
    #H2_linear_score = H1_ReLU_score.dot(W2)    #shape (N,C)
    #print 'H2_linear_score shape:', H2_linear_score.shape   #should be (N,C)

    #output layer. without activation.
    scores = h1.dot(W2) + b2      #shape (N,C)

    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################
    
    # If the targets are not given then jump out, we're done
    if y is None:
      return scores

    # Compute the loss
    loss = None
    #############################################################################
    # TODO: Finish the forward pass, and compute the loss. This should include  #
    # both the data loss and L2 regularization for W1 and W2. Store the result  #
    # in the variable loss, which should be a scalar. Use the Softmax           #
    # classifier loss. So that your results match ours, multiply the            #
    # regularization loss by 0.5                                                #
    #############################################################################
    pass
    #by kenneth yu

    #calc the soft max loss. shift scores.....
    #refer to softmax.py
    #shift scores by max of each row(each sample scores). reshape to avoid ambiguity of broadcasting
    #in case N=C ....
    scores -= np.max(scores, axis=-1).reshape((scores.shape[0], -1))

    #sum_p each row corresponds to each shifted sample scores.
    sum_p = np.sum(np.exp(scores),axis=-1)   #sum_p shape(num_train, )

    loss = - np.sum(scores[np.array(xrange(scores.shape[0])), y]) + np.sum(np.log(sum_p))

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= X.shape[0]
    #. average gradient.
    #dW /= num_train

    # Add regularization to the loss.elementwise product.
    # dont forget the bias.
    loss += 0.5 * reg * (np.sum(W1 * W1) + np.sum(b1 * b1)
                         + np.sum(W2 * W2) + np.sum(b2 * b2))

    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    # Backward pass: compute gradients
    grads = {}
    #############################################################################
    # TODO: Compute the backward pass, computing the derivatives of the weights #
    # and biases. Store the results in the grads dictionary. For example,       #
    # grads['W1'] should store the gradient on W1, and be a matrix of same size #
    #############################################################################
    pass
    #kenneth yu. refer to classifier/softmax.
    ### BP ###
    #denotion:  dxyz = d(Full loss)/d(xyz)

    #d( Full loss )/ d( output score of each class w.r.t each data example )
    #backprop Full loss = -exp(score(yi)......
    # mark the y[i]
    Indicator = np.zeros_like(scores)       #shape (N,C)
    Indicator[np.array(xrange(Indicator.shape[0])), y] += -1

    # mark the derivative of logs
    d_log = np.divide(np.exp(scores), sum_p.reshape(sum_p.shape[0],-1))        #shape (N,C)

    doutput_scores = Indicator + d_log  #shape (N,C)


    #d(Full loss ) / d( output layer - b2, W2).
    #backprop: output_scores = W2.dot(h1) + b2.
    # sum the same bias along all the data examles. will average in the end of SGD
    db2 = np.sum(doutput_scores, axis=0)  #shape (C,)

    #h1 shape (N,H)
    dW2 = h1.T.dot(doutput_scores)      #shape (H,C)
    dh1 = doutput_scores.dot(W2.T)      #shape (N,H)

    #d(Full loss) / d( H-1 layer - ReLU)
    #backprop: h1 = ReLU(linear output of H1)
    bp_activation_f = lambda x: (x > 0).astype('uint8') #ReLU
    dlinear_output_H1 = bp_activation_f(h1) * dh1 #shape (N,H) * (N,H) elementwise product.

    #d(Full loss) / d( H-1 layer - b1, W1)
    #backprop:  linear_output_H1 = W1.dot(x) + b1
    db1 = np.sum(dlinear_output_H1, axis=0)
    #X shape (N,D)
    dW1 = X.T.dot(dlinear_output_H1)            #shape (D,H)

    #average gradient.
    average_on_examples = lambda x: x/X.shape[0]
    dW1 = average_on_examples(dW1)
    db1 = average_on_examples(db1)
    dW2 = average_on_examples(dW2)
    db2 = average_on_examples(db2)
    ### finish of the score loss gradient.####

    ### kenneth yu. add the regu loss gradient.###
    #dW +=  0.5 * 2 * reg * W
    #refer to: regu loss = 0.5 * reg * (np.sum(W1 * W1) + np.sum(b1 * b1)
    #                       + np.sum(W2 * W2) + np.sum(b2 * b2))

    regu_gradient = lambda x: reg * x
    dW1 += regu_gradient(W1)
    db1 += regu_gradient(b1)
    dW2 += regu_gradient(W2)
    db2 += regu_gradient(b2)

    #save the result to dictionary.
    grads['W1'] = dW1
    grads['b1'] = db1
    grads['W2'] = dW2
    grads['b2'] = db2

    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    return loss, grads

  def train(self, X, y, X_val, y_val,
            learning_rate=1e-3, learning_rate_decay=0.95,
            reg=1e-5, num_iters=100,
            batch_size=200, verbose=False):
    """
    Train this neural network using stochastic gradient descent.

    Inputs:
    - X: A numpy array of shape (N, D) giving training data.
    - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
      X[i] has label c, where 0 <= c < C.
    - X_val: A numpy array of shape (N_val, D) giving validation data.
    - y_val: A numpy array of shape (N_val,) giving validation labels.
    - learning_rate: Scalar giving learning rate for optimization.
    - learning_rate_decay: Scalar giving factor used to decay the learning rate
      after each epoch.
    - reg: Scalar giving regularization strength.
    - num_iters: Number of steps to take when optimizing.
    - batch_size: Number of training examples to use per step.
    - verbose: boolean; if true print progress during optimization.
    """
    num_train = X.shape[0]
    iterations_per_epoch = max(num_train / batch_size, 1)

    # Use SGD to optimize the parameters in self.model
    loss_history = []
    train_acc_history = []
    val_acc_history = []

    for it in xrange(num_iters):
      X_batch = None
      y_batch = None

      #########################################################################
      # TODO: Create a random minibatch of training data and labels, storing  #
      # them in X_batch and y_batch respectively.                             #
      #########################################################################
      pass
      #kenneth yu
      batch_idx = np.random.choice(num_train, size=batch_size, replace=True)
      #print 'batch idx:', batch_idx
      X_batch = X[batch_idx, :]      # X shape:(N,D). X_batch shape:(batch_size,D)
      y_batch = y[batch_idx]         # y shape:(N,)

      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################

      # Compute loss and gradients using the current minibatch
      loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
      loss_history.append(loss)

      #########################################################################
      # TODO: Use the gradients in the grads dictionary to update the         #
      # parameters of the network (stored in the dictionary self.params)      #
      # using stochastic gradient descent SGD. You'll need to use the gradients   #
      # stored in the grads dictionary defined above.                         #
      #########################################################################
      #kenneth yu
      self.params['W1'] += -learning_rate * grads['W1']
      self.params['b1'] += -learning_rate * grads['b1']
      self.params['W2'] += -learning_rate * grads['W2']
      self.params['b2'] += -learning_rate * grads['b2']

      pass
      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################

      if verbose and it % 100 == 0:
        print 'iteration %d / %d: loss %f' % (it, num_iters, loss)

      # Every epoch, check train and val accuracy and decay learning rate.
      if it % iterations_per_epoch == 0:
        # Check accuracy
        train_acc = (self.predict(X_batch) == y_batch).mean()
        val_acc = (self.predict(X_val) == y_val).mean()
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)

        # Decay learning rate
        learning_rate *= learning_rate_decay

    return {
      'loss_history': loss_history,
      'train_acc_history': train_acc_history,
      'val_acc_history': val_acc_history,
    }

  def predict(self, X):
    """
    Use the trained weights of this two-layer network to predict labels for
    data points. For each data point we predict scores for each of the C
    classes, and assign each data point to the class with the highest score.

    Inputs:
    - X: A numpy array of shape (N, D) giving N D-dimensional data points to
      classify.

    Returns:
    - y_pred: A numpy array of shape (N,) giving predicted labels for each of
      the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
      to have class c, where 0 <= c < C.
    """
    y_pred = None

    ###########################################################################
    # TODO: Implement this function; it should be VERY simple!                #
    ###########################################################################
    pass
    #kenneth yu
    #define activation function. ReLU
    activation_f = lambda x: (x > 0).astype('uint8') * x
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']

    #forward prop : h1 = ReLU(W1 dot X + b1)
    h1 =activation_f( (X.dot(W1) + b1) )

    #fp:  scores = W2 dot h1 + b2
    scores = h1.dot(W2) + b2    #  scores shape (N,C)

    y_pred = np.argmax(scores,axis=-1)  #y_pred shape (N,)


    ###########################################################################
    #                              END OF YOUR CODE                           #
    ###########################################################################

    return y_pred


