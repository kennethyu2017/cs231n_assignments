import numpy as np
import matplotlib.pyplot as plt

from cs231n.layers import *
from cs231n.layer_utils import *



class TwoLayerNet(object):
  """
  A two-layer fully-connected neural network with ReLU nonlinearity and
  softmax loss that uses a modular layer design. We assume an input dimension
  of D, a hidden dimension of H, and perform classification over C classes.
  
  The architecure should be affine - relu - affine - softmax.

  Note that this class does not implement gradient descent; instead, it
  will interact with a separate Solver object that is responsible for running
  optimization.

  The learnable parameters of the model are stored in the dictionary
  self.params that maps parameter names to numpy arrays.
  """
  
  def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,drop_out=0.5,
               weight_scale=1e-3, reg=0.0, loss_func = 'softmax'):
    """
    Initialize a new network.

    Inputs:
    - input_dim: An integer giving the size of the input
    - hidden_dim: An integer giving the size of the hidden layer
    - num_classes: An integer giving the number of classes to classify
    - dropout: Scalar between 0 and 1 giving dropout strength.
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - reg: Scalar giving L2 regularization strength.
    """
    self.params = {}
    self.reg = reg
    self.loss_func = loss_func

    #added by kenneth yu.
    self.drop_out = drop_out
    
    ############################################################################
    # TODO: Initialize the weights and biases of the two-layer net. Weights    #
    # should be initialized from a Gaussian with standard deviation equal to   #
    # weight_scale, and biases should be initialized to zero. All weights and  #
    # biases should be stored in the dictionary self.params, with first layer  #
    # weights and biases using the keys 'W1' and 'b1' and second layer weights #
    # and biases using the keys 'W2' and 'b2'.                                 #
    ############################################################################

    pass
    # kenneth yu


    # init weights with calibration by He.
    # use double precision number. shape (i_dim, o_dim).
    init_weights = lambda i_dim,o_dim: np.random.randn(i_dim, o_dim) * np.sqrt(2.0/(i_dim))
    #not use calibration at first toy data test.

    # !!!!!!!!!!!!!!!!!!!     Note: use double pecision as possible.      !!!!!!!!!!!!!!!!!!!!!
    # init_weights = lambda i_dim,o_dim: np.float64(weight_scale) * np.random.randn(i_dim, o_dim)

    # init bias to zero. shape(1, o_dim)
    init_bias = lambda o_dim : np.zeros(o_dim, dtype=np.float64)


    # init 1st layer params:
    self.params['W1'] = init_weights(input_dim, hidden_dim)  #shape (D,H1)
    self.params['b1'] = init_bias(hidden_dim)                #shape (H1,)

    print "1st layer weights init std:", self.params['W1'].std()

    # init 2nd layer params: output layer. w/o relu.
    self.params['W2'] = init_weights(hidden_dim,num_classes)  #shape (H1,C)
    self.params['b2'] = init_bias(num_classes)                #shape (H1,)


    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################


  def loss(self, X, y=None):
    """
    Compute loss and gradient for a minibatch of data.

    Inputs:
    - X: Array of input data of shape (N, d_1, ..., d_k)
    - y: Array of labels, of shape (N,). y[i] gives the label for X[i].
    - loss_func: 'softmax': use cross-entropy loss.
                  'svm': use svm hinge loss.

    Returns:
    If y is None, then run a test-time forward pass of the model and return:
    - scores: Array of shape (N, C) giving classification scores, where
      scores[i, c] is the classification score for X[i] and class c.

    If y is not None, then run a training-time forward and backward pass and
    return a tuple of:
    - loss: Scalar value giving the loss
    - grads: Dictionary with the same keys as self.params, mapping parameter
      names to gradients of the loss with respect to those parameters.
    """  
    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the two-layer net, computing the    #
    # class scores for X and storing them in the scores variable.              #
    ############################################################################
    pass
    #kenneth yu

    # calc the FP of H-1 . with relu.
    # improve X to double precision as possible.
    h1_out, h1_cache = affine_relu_forward(X.astype(np.float64), self.params['W1'], self.params['b1'])

    #calc the FP of H-2. output layer w/o relu.
    h2_out, h2_cache = affine_forward(h1_out, self.params['W2'], self.params['b2'])

    scores = h2_out  #shape (N, C)

	############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # If y is None then we are in test mode so just return scores
    if y is None:
		return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the two-layer net. Store the loss  #
    # in the loss variable and gradients in the grads dictionary. Compute data #
    # loss using softmax, and make sure that grads[k] holds the gradients for  #
    # self.params[k]. Don't forget to add L2 regularization!                   #
    #                                                                          #
    # NOTE: To ensure that your implementation matches ours and you pass the   #
    # automated tests, make sure that your L2 regularization includes a factor #
    # of 0.5 to simplify the expression for the gradient.                      #
    ############################################################################
    pass
    # kenneth yu
    # calc loss. default is softmax.
    # if scores is double precision , then the dscores, data_loss are double precision also.
    if self.loss_func == 'softmax':
      data_loss, dscores = softmax_loss(scores, y)
    else:
      data_loss, dscores = svm_loss(scores, y)

    #no need to do bias reg.
    l2_norm = lambda w : np.linalg.norm(w) ** 2
    reg_loss = (1.0/2) * self.reg * ( l2_norm(self.params['W1']) + l2_norm(self.params['W2']))

    loss = data_loss + reg_loss

    #calc BP grad
    #calc BP of H-2 ,the output layer.
    dh1_output, grads['W2'], grads['b2'] = affine_backward(dscores, h2_cache)

    # print ' W2 value ==>', self.params['W2'][28,:3]
    # print ' grad before reg ==>',  grads['W2'][28,:3]
    # #add the reg items. no reg on bias.
    grads['W2'] += self.reg * self.params['W2']

    # print ' grad after reg ==>',  grads['W2'][28,:3]



    #calc BP of H-1.
    dinput, grads['W1'], grads['b1'] = affine_relu_backward(dh1_output, h1_cache)
    grads['W1'] += self.reg * self.params['W1']


    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads


class FullyConnectedNet(object):
		"""
		A fully-connected neural network with an arbitrary number of hidden layers,
		ReLU nonlinearities, and a softmax loss function. This will also implement
		dropout and batch normalization as options. For a network with L layers,
		the architecture will be

		{affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax

		where batch normalization and dropout are optional, and the {...} block is
		repeated L - 1 times.

		Similar to the TwoLayerNet above, learnable parameters are stored in the
		self.params dictionary and will be learned using the Solver class.
		"""

		def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
							 dropout=0, use_batchnorm=False, reg=0.0,
							 weight_scale=1e-2, dtype=np.float32, seed=None):

		# Initialize a new FullyConnectedNet.
		#
		# Inputs:
		# - hidden_dims: A list of integers giving the size of each hidden layer.
		# - input_dim: An integer giving the size of the input.
		# - num_classes: An integer giving the number of classes to classify.
		# - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
		# 	the network should not use dropout at all.
		# - use_batchnorm: Whether or not the network should use batch normalization.
		# - reg: Scalar giving L2 regularization strength.
		# - weight_scale: Scalar giving the standard deviation for random
		# 	initialization of the weights.
		# - dtype: A numpy datatype object; all computations will be performed using
		# 	this datatype. float32 is faster but less accurate,
		#
		# 	!!!! so you should use float64 for numeric gradient checking. !!!!!!!
		#
		# - seed: If not None, then pass this random seed to the dropout layers. This
		# 	will make the dropout layers deteriminstic so we can gradient check the
		# 	model.


			self.use_batchnorm = use_batchnorm
			self.use_dropout = dropout > 0
			self.reg = reg
			self.num_layers = 1 + len(hidden_dims)  # plus the output layer
			self.dtype = dtype
			self.params = {}

			############################################################################
			# TODO: Initialize the parameters of the network, storing all values in    #
			# the self.params dictionary. Store weights and biases for the first layer #
			# in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
			# initialized from a normal distribution with standard deviation equal to  #
			# weight_scale and biases should be initialized to zero.                   #
			#                                                                          #
			# When using batch normalization, store scale and shift parameters for the #
			# first layer in gamma1 and beta1; for the second layer use gamma2 and     #
			# beta2, etc. Scale parameters should be initialized to one and shift      #
			# parameters should be initialized to zero.                                #
			############################################################################
			pass
			# kenneth yu
			# optional: init weights with calibration by He.
			# use double precision number. shape (i_dim, o_dim)
			# Note: we only calibrate with the i_dim, i.e. the input dimension of a single neuron.
			# !!!!! and call randn init one single neuron one time!!! dont use one single randn call
			#	!!!!!! to init all the weight vectors of a layer !!!!!!!!!!!
			# !!!! but after test , the result very close when i_dim=3072, o_dim=100. but in !!!
			# !!!! case the i_dim , o_dim are small(e.g. <10 ) , the mean of weight vector of
			# each neuron will be a bit away from zero , and cause the affine output will be
			# all negative or all positive.
			# init_weights = lambda i_dim,o_dim: np.array( [ np.random.randn(i_dim).astype(dtype)\
			# 																	 * np.sqrt(2.0/(i_dim)) for i in xrange(o_dim)] ).T

			init_weights = lambda i_dim,o_dim: np.random.randn(i_dim, o_dim).astype(dtype)\
			 																	 * np.sqrt(2.0/(i_dim))


			#	init func
			# !!     Note: for grad check, use double pecision as possible.      !!
			# init_weights = lambda i_dim,o_dim: weight_scale * np.random.randn(i_dim, o_dim).astype(dtype)


			# bias init func.to zero. shape(1, o_dim)
			init_bias = lambda o_dim : np.zeros(o_dim, dtype=dtype)

			# init gamma, beta for bn layer.
			init_gamma = lambda o_dim: np.ones(o_dim, dtype=dtype)
			init_beta = lambda o_dim: np.zeros(o_dim,dtype=dtype)

			# init the hidden layers
			# for convinience, we insert the input size and class num into the list.
			layer_dims = np.hstack((input_dim, hidden_dims, num_classes))


			for l in np.arange(len(layer_dims) - 1):
				# when l is 0, the layer_dim is input size.
				previous_layer_dim = layer_dims[l]
				this_layer_dim = layer_dims[l + 1]
				self.params['W' + str(l + 1)] = init_weights(previous_layer_dim, this_layer_dim)
				self.params['b' + str(l + 1)] = init_bias(this_layer_dim)
				# print "FCNet init == > %d layer weights init std: %e" % (l+1,self.params['W' + str(l+1)].std())

				#	last output layer not have bn params.
				if use_batchnorm and (l+1)!= self.num_layers:  # self.num_layers include output layer.
					self.params['gamma' + str(l + 1)] = init_gamma(this_layer_dim)
					self.params['beta' + str(l + 1)] = init_beta(this_layer_dim)

			############################################################################
			#                             END OF YOUR CODE                             #
			############################################################################

			# When using dropout we need to pass a dropout_param dictionary to each
			# dropout layer so that the layer knows the dropout probability and the mode
			# (train / test). You can pass the SAME dropout_param
			#  to EACH dropout layer.
				self.dropout_param = {}
				if self.use_dropout:
					self.dropout_param = {'mode': 'train', 'p': dropout}
				if seed is not None:
					#	seed for grad check. same for all the hidden layers.
					self.dropout_param['seed'] = seed

			# With batch normalization we need to keep track of running means and
			# variances, so we need to pass a special bn_param object to each batch
			# normalization layer. You should pass self.bn_params[0] to the forward pass
			# of the first batch normalization layer, self.bn_params[1] to the forward
			# pass of the second batch normalization layer, etc.
			# mean and var running average belong to model parameters for test.
			# bn_param: list of Dictionary with the following keys:
			# - mode: 'train' or 'test'; required
			# - eps: Constant for numeric stability
			# - momentum: Constant for running mean / variance.
			# - running_mean: Array of shape (D,) giving running mean of features
			# - running_var Array of shape (D,) giving running variance of features

			self.bn_params = []
			if self.use_batchnorm:
				#	not include last output layer
				self.bn_params = [{'mode': 'train'} for i in xrange(self.num_layers - 1)]
				# !!!! error:  duplicate dict address !!!
				# self.bn_params = [{'mode': 'train'}] * (self.num_layers - 1)

			# Cast all parameters to the correct datatype
			for k, v in self.params.iteritems():
				self.params[k] = v.astype(dtype)


		def loss(self, X, y=None, layer_verbose=False,layer_plot=False, bp=True):
			"""

			Compute loss and gradient for the fully-connected net.

			Input / output: Same as TwoLayerNet above.
			"""
			#	transfer the data type to dtype. careful about precision.
			global out
			X = X.astype(self.dtype)
			mode = 'test' if y is None else 'train'

			# Set train/test mode for batchnorm params and dropout param since they
			# behave differently during training and testing.
			if self.dropout_param is not None:
				self.dropout_param['mode'] = mode
			if self.use_batchnorm:
				for bn_param in self.bn_params:
					bn_param[mode] = mode

			scores = None
			############################################################################
			# TODO: Implement the forward pass for the fully-connected net, computing  #
			# the class scores for X and storing them in the scores variable.          #
			#                                                                          #
			# When using dropout, you'll need to pass self.dropout_param to each       #
			# dropout forward pass.                                                    #
			#                                                                          #
			# When using batch normalization, you'll need to pass self.bn_params[0] to #
			# the forward pass for the first batch normalization layer, pass           #
			# self.bn_params[1] to the forward pass for the second batch normalization #
			# layer, etc.                                                              #
			############################################################################
			pass
			#	kenneth yu
			#	??????? data pre processing ??????

			# if layer_verbose:
			# 	print '\n Training ===  layer  FP BP data >>>>>'

			#	record of cache of each layer for the usage of BP.
			# should be clean for each batch of data.
			#	index by (layer_number)
			layer_caches = {}

			# calc the output of each layer, and record the cache.
			#	input data.
			previous_out = X.reshape(X.shape[0], -1)			#	shape (N, d_1,d_2, ..., d_k)

			layer_params = {}
			for l in np.arange(start=1, stop=self.num_layers+1):
				#	num_layers include last output layer.
				# arange excludes the stop value.
				layer_params['W'] = self.params['W' + str(l)]
				layer_params['b'] = self.params['b'	+	str(l)]

				is_output_layer = True if (l == self.num_layers) else False

				if self.use_batchnorm and not is_output_layer:
					layer_params['gamma'] = self.params['gamma' + str(l)]
					layer_params['beta'] = self.params['beta' + str(l)]
					#	self.bn_params is a list of Dictionary.
					layer_params['bn_param'] = self.bn_params[l-1]

				if self.use_dropout and not is_output_layer:
					# same for all the hidden layers.
					layer_params['dropout'] = self.dropout_param

				# if layer_verbose:
					# print '==>>>> %% layer %d input ' % l, previous_out
					# print ' === %% layer %d params W and b: ' % l, layer_params['W'], layer_params['b']
					# print ' === %% layer %d params W total mean %e: ' % (l, np.mean(layer_params['W']))
					# print ' === %% layer %d mean of W of each node :' % (l), np.mean(layer_params['W'], axis=0),\
					# 								' var of W of each node :', np.var(layer_params['W'], axis=0)

				if is_output_layer:
					#	output layer w/o ReLU/BN/dropout...only affine.
					scores, cache = affine_forward(previous_out,layer_params['W'], layer_params['b'])
				else:
					# if	self.use_batchnorm and layer_verbose:
					# 	if layer_params['bn_param'].has_key('running_mean'):
					# 		print ' ==== layer %d bn_param running mean before:' % ( l), layer_params['bn_param']['running_mean']
					# 	if layer_params['bn_param'].has_key('running_var'):
					# 		print ' ==== layer %d bn_param running var before:' % ( l),  layer_params['bn_param']['running_var']
					previous_out, cache = hidden_forward(previous_out, layer_params,
																			use_batchnorm=self.use_batchnorm,
																			use_dropout=self.use_dropout)

				# ??? need a copy of cache ???#
				layer_caches[str(l)] = cache

				if layer_verbose:
					# print ' %% layer %d out mean : %e  variance: %e' %  (l, np.mean(out), np.var(out))
					print ' <<<#####  %% layer %d out  ' % l,  scores if is_output_layer else previous_out

					# if	self.use_batchnorm and layer_verbose:
					# 	print ' ==== layer %d bn_param running mean after:' %(l), layer_params['bn_param']['running_mean']
					# 	print ' ==== layer %d bn_param running var after:' % (l), layer_params['bn_param']['running_var']

					# ^^^^  for num grad check error of bias debug ^^^^^^###
					# out, cache = affine_forward(previous_out, W, b)

					# if l == 4:
					# 	print ' && layer 4 fc cache x=>', cache[0][0]
					# 	print ' && layer 4 fc cache w=>', cache[0][1]
					# 	print ' && layer 4 fc cache b=>', cache[0][2]
					# 	print ' && layer 4 relu cache =>', cache[1]

				# if layer_plot:
					# plt.subplot(1, self.num_layers, l)
					# plt.hist(out.ravel(), bins=200)

			# if layer_verbose:
			# 	# plt.gcf().set_size_inches(15,10)
			# 	# plt.show()

			###### end of layer fp loop ######

			############################################################################
			#                             END OF YOUR CODE                             #
			############################################################################
			# If test mode return early
			if mode == 'test':
				return scores

			#	grads for each batch of data. should be clean for each batch.
			loss, grads = 0.0, {}
			############################################################################
			# TODO: Implement the backward pass for the fully-connected net. Store the #
			# loss in the loss variable and gradients in the grads dictionary. Compute #
			# data loss using softmax, and make sure that grads[k] holds the gradients #
			# for self.params[k]. Don't forget to add L2 regularization!               #
			#                                                                          #
			# When using batch normalization, you don't need to regularize the scale   #
			# and shift parameters.                                                    #
			#                                                                          #
			# NOTE: To ensure that your implementation matches ours and you pass the   #
			# automated tests, make sure that your L2 regularization includes a factor #
			# of 0.5 to simplify the expression for the gradient.                      #
			############################################################################
			pass
			#	kenneth yu
			# calc loss. default is softmax.
			# if scores is double precision , then the dscores, data_loss are double precision also.

			data_loss, dscores = softmax_loss(scores, y)

			if layer_verbose:
					print ' ==> dscore mean : %e  variance: %e' %  ( np.mean(dscores), np.var(dscores))

			# no need to do bias reg.
			l2_norm = lambda w : np.linalg.norm(w) ** 2
			reg_loss = self.dtype(0)
			#	sum l2_norm of each layer. No reg for bias.
			#	Dont forget the last layer with care about the stop value.
			for l in np.arange(start=1, stop=self.num_layers+1):
				reg_loss += (1.0/2) * self.reg * l2_norm(self.params['W' + str(l)])

			loss = data_loss + reg_loss

			if bp is False:
				return loss

			# calc BP grad.
			# the dout for output layer is just the dscores.
			next_layer_dout = dscores

			for l in np.arange(start=self.num_layers, stop=0, step=-1):
				#	get cache of the 'l' layer. pop to reduce memory usage.
				cache =	layer_caches.pop(str(l))
				# cache =	layer_caches.get(str(l))

				# print 'layer %d cache ==>' %(l)
				# print cache

				w_name = 'W' + str(l)
				b_name = 'b' + str(l)

				is_output_layer = True if (l == self.num_layers) else False

				if self.use_batchnorm and not is_output_layer:
					gamma_name = 'gamma' + str(l)
					beta_name = 'beta' + str(l)
					#	self.bn_params is a list of Dictionary.
					layer_params['bn_param'] = self.bn_params[l-1]

				if is_output_layer:
					# w/o ReLU
					next_layer_dout, grads[w_name], grads[b_name] = affine_backward(next_layer_dout, cache)
					# if b_name == 'b5':
					# 	print '@@  dscores: ' , next_layer_dout
				  #  	print 'grads  %s ==>' % ( b_name )
					# 	print grads[b_name]
				else:
					next_layer_dout, d_params = hidden_backward(next_layer_dout, cache)

					# affine->bn->relu->dropout . check reversely.
					if self.bn_params:
						grads[gamma_name] = d_params.get('bn').get('gamma')
						grads[beta_name] = d_params.get('bn').get('bata')

					grads[w_name] = d_params.get('affine').get('w')
					grads[b_name] = d_params.get('affine').get('b')

					# ^^^^  for num grad check error of bias debug ^^^^^^###
					# dx, grads[w_name], grads[b_name] = affine_backward(next_layer_dout, cache)

				# print 'grads  %s ==>' % ( b_name )
				# print grads[b_name]

				# print ' %s weight value [10,3] ==> %e' %(w_name,  self.params[w_name][10,3] )
				# print ' %s grad [10][3] before reg  ==> %e' % (w_name, grads[w_name][10,3])
				# #plus the reg grad. no reg on bias
				# reg on gamma?????
				grads[w_name] += self.reg * self.params[w_name]

				if layer_verbose:
					print ' %% layer %d grads W mean : %e  variance: %e' %  (l, np.mean(grads[w_name]), np.var(grads[w_name]))

				# if l == 3:
				# 	print 'grads %s after reg ==>'%( w_name)
				# 	print grads[w_name]
				# 	print 'W3: ==>' ,self.params[w_name]

				# print ' grad[10][3] after reg ==> %e' % ( grads[w_name][10,3])

			############################################################################
			#                             END OF YOUR CODE                             #
			############################################################################
			return loss, grads


# def output_forward(x, layer_params):
# 	W = layer_params.get('W')
# 	b = layer_params.get('b')
#
# 	cache = {}
# 	out, cache['affine'] = affine_forward(x, W, b)
# 	return out, cache


def hidden_forward(x, layer_params, use_batchnorm=False, use_dropout=False):
	# input :
	# - params: dict of layer params.keys:
	# 			- 'W','b', 'gamma', 'beta'
	# outpu:
	#	- cache: dict of each unit cache.
	W = layer_params.get('W', None)
	b = layer_params.get('b', None)

	assert W is not None and b is not None
	out = []
	cache = {}
	node_out = x

	### nodes:  affine -> bn -> relu -> dropout  ###

	# affine
	node_out, cache['affine'] = affine_forward(node_out, W, b)
	# print '===== out of affine: ', node_out
	# print '===== out of affine mean: ', np.mean(node_out)


	# bn
	if layer_params.has_key('gamma'):
		gamma = layer_params.get('gamma', None)
		beta = layer_params.get('beta', None)
		bn_param = layer_params.get('bn_param', None)
		assert gamma is not None and beta is not None and bn_param is not None
		# out,cache['bn'] = affine_bn_relu_forward(x, W=W, b=b, gamma=gamma, beta=beta, bn_param=bn_param)
		# print '== out of affine: ', af_out
  	# bn
		node_out, cache['bn'] = batchnorm_forward(node_out, gamma, beta, bn_param)

	# relu
	# print ' +++++ input to relu: ', node_out
	node_out, cache['relu'] = relu_forward(node_out)
	# print '== out of relu: ', node_out
	# relu

	#	dropout
	if layer_params.has_key('dropout'):
		assert layer_params['dropout'] is not None
		node_out, cache['dropout'] = dropout_forward(node_out,layer_params.get('dropout'))
		# print ' ==> out of dropout:  ', node_out
		# print ' ==>  dropout output non-zero ratios over each sample : ', np.mean(node_out!=0, axis=1)


	out = node_out
	return out, cache



def hidden_backward(dout, cache, use_batchnorm=False, use_dropout=False):
	# input :
	# - dout
	# - cache: dict
	#
	# output:
	#	- grads: dict.
	#

	dx = dout
	d_params = {}

	# node FP sequence:  affine -> bn -> relue -> dropout

	# BP start from dropout
	if cache.has_key('dropout'):
		dropout_cache = cache.get('dropout', None)
		assert dropout_cache is not None
		#	no parameter then no grads of dropout
		dx = dropout_backward(dx, dropout_cache)

	#	relu. no parameter, so no grads.
	dx = relu_backward(dx, cache.get('relu'))

	# bn
	if cache.has_key('bn'):
		d_params['bn'] = {}
		dx, d_params['bn']['gamma'], d_params['bn']['beta'] = batchnorm_backward_alt(dx, cache.get('bn'))

	# affine
	d_params['affine'] = {}
	dx, d_params['affine']['w'], d_params['affine']['b'] = affine_backward(dx, cache.get('affine'))

	return dx, d_params
